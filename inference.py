import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os

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

def load_model(checkpoint_path, device, img_size=128, patch_size=16, in_chans=3, embed_dim=1024, num_heads=16, num_layers=12):
    model = ImageUpscaler(img_size, patch_size, in_chans, embed_dim, num_heads, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handling the DDP "module." prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def infer(model, image_path, device, output_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize low-resolution images to 128x128 pixels
        transforms.ToTensor()
    ])
    
    # Load and transform the input image
    lr_image = Image.open(image_path).convert('RGB')
    lr_image = transform(lr_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        hr_image = model(lr_image).squeeze(0).cpu()

    # Convert the output tensor to an image
    hr_image = transforms.ToPILImage()(hr_image)
    hr_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Inference script for super-resolution model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to the low-resolution input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the high-resolution output image')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    args = parser.parse_args()

    # Ensure the embedding dimension is divisible by the number of heads
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by the number of attention heads")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(args.checkpoint, device, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers)
    
    # Perform inference
    infer(model, args.input, device, args.output)
    
if __name__ == "__main__":
    main()
