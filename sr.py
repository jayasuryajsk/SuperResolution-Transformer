import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image
import zipfile
import requests
from tqdm import tqdm

def download_dataset(url, save_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print("Extracting dataset...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(save_path)  # Remove the zip file after extraction
        print("Dataset ready.")

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, downscale_factor=4):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.downscale_factor = downscale_factor
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert('RGB')
        lr_image = hr_image.resize((hr_image.width // self.downscale_factor, hr_image.height // self.downscale_factor), Image.BICUBIC)
        
        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.target_transform(hr_image)
        
        return lr_image, hr_image

class ImageTokenizer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, grid_size*grid_size)
        x = x.transpose(1, 2)  # (B, grid_size*grid_size, embed_dim)
        return x

class ImageDeTokenizer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x.view(x.shape[0], -1, self.grid_size, self.grid_size)  # (B, embed_dim, grid_size, grid_size)
        x = self.proj(x)  # (B, in_chans, img_size, img_size)
        return x

class UpscaleLayer(nn.Module):
    def __init__(self, scale_factor=8):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.upscale(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x, memory):
        x = x + self.attn1(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.attn2(self.norm2(x), memory, memory)[0]
        x = x + self.mlp(self.norm3(x))
        return x

class ImageUpscaler(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=1024, num_heads=16, num_layers=12):
        super().__init__()
        self.tokenizer = ImageTokenizer(img_size, patch_size, in_chans, embed_dim)
        self.de_tokenizer = ImageDeTokenizer(img_size, patch_size, in_chans, embed_dim)
        self.upscale_layer = UpscaleLayer(scale_factor=8)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokenizer.num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads)
        for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.tokenizer(x) + self.pos_embed
        memory = x.clone()
        for block in self.transformer_blocks:
            x = block(x, memory)
        x = self.norm(x)
        x = self.de_tokenizer(x)
        x = self.upscale_layer(x)
        return x

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize low-resolution images to 128x128 pixels
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Resize high-resolution images to 1024x1024 pixels
        transforms.ToTensor()
    ])

    dataset = DIV2KDataset(args.dataset_path, transform=transform, target_transform=target_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    if len(data_loader) == 0:
        raise ValueError("DataLoader is empty. Ensure the dataset path is correct and contains images.")

    model = ImageUpscaler(img_size=128, patch_size=16, in_chans=3, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers).cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for lr_images, hr_images in data_loader:
            lr_images, hr_images = lr_images.cuda(rank, non_blocking=True), hr_images.cuda(rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f'Rank {rank}, Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(data_loader):.4f}')
        
        # Save the model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            if rank == 0:  # Ensure only one process saves the checkpoint
                checkpoint_path = f"model_epoch_{epoch+1}.pth"
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss / len(data_loader),
                }, filename=checkpoint_path)

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train a super-resolution model using DDP.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu_count', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')

    args = parser.parse_args()

    # Validate that embed_dim is divisible by num_heads
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by the number of attention heads")

    # Define dataset URL
    dataset_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
    
    # Get current working directory and set dataset path
    current_working_directory = os.getcwd()
    dataset_path = os.path.join(current_working_directory, 'DIV2K_train_HR')

    # Handle nested directory if it exists
    if os.path.exists(os.path.join(dataset_path, 'DIV2K_train_HR')):
        dataset_path = os.path.join(dataset_path, 'DIV2K_train_HR')

    # Ensure the dataset path exists and is not empty
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        download_dataset(dataset_url, dataset_path + '.zip', dataset_path)

    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Update args to include dataset_path
    args.dataset_path = dataset_path

    mp.spawn(train,
             args=(args.gpu_count, args),
             nprocs=args.gpu_count,
             join=True)

if __name__ == "__main__":
    main()
