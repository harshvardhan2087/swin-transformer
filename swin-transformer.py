import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def window_partition(x, win):
    B, H, W, C = x.shape
    x = x.view(B, H // win, win, W // win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, win, win, C)

def window_reverse(windows, win, H, W):
    Nw = windows.shape[0]
    B = int(Nw // ((H // win) * (W // win)))
    C = windows.shape[-1]
    x = windows.view(B, H // win, W // win, win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, win):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win = win
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij"))
        coords_flat = coords.view(2, -1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0)
        rel[:, :, 0] += win - 1
        rel[:, :, 1] += win - 1
        rel[:, :, 0] *= 2 * win - 1
        index = rel.sum(-1)
        self.register_buffer("relative_index", index.long())
        self.relative_bias = nn.Parameter(torch.zeros((2 * win - 1) ** 2, num_heads))

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rb = self.relative_bias[self.relative_index.view(-1)]
        rb = rb.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + rb

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, res, win, shift, heads):
        super().__init__()
        self.res = res
        self.win = win
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, win)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        if shift > 0:
            self.register_buffer("mask", self.create_mask(res[0], res[1], win, shift))
        else:
            self.register_buffer("mask", None)

    def create_mask(self, H, W, win, shift):
        img_mask = torch.zeros((1, H, W, 1))
        count = 0
        for h in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
            for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
                img_mask[:, h, w, :] = count
                count += 1
        mask = window_partition(img_mask, win).view(-1, win * win)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.masked_fill(mask != 0, -10000.0)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.res
        residual = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))

        x = window_partition(x, self.win).view(-1, self.win * self.win, C)
        x = self.attn(x, self.mask.to(x.device) if self.mask is not None else None)
        x = window_reverse(x, self.win, H, W)

        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))

        x = x.view(B, L, C) + residual
        return x + self.mlp(self.norm2(x))

class TwoStageSwinMNIST(nn.Module):
    def __init__(self, embed_dim=48, heads=3, win=7):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, 2, 2)
        self.blocks = nn.Sequential(
            SwinBlock(embed_dim, (14, 14), win, 0, heads),
            SwinBlock(embed_dim, (14, 14), win, win // 2, heads)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 10)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x).mean(1)
        return self.fc(x)

def visualize_predictions(model, val_loader, device):
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)

    model = TwoStageSwinMNIST().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            loss = loss_fn(model(img), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()
