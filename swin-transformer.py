# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# def window_partition(x, win):
#     B,H,W,C = x.shape
#     x = x.view(B, H//win, win, W//win, win, C)
#     x = x.permute(0,1,3,2,4,5)
#     x = x.reshape(-1, win, win, C)
#     return x

# def window_reverse(windows, win, H, W):
#     B = windows.shape[0]/(H//win * W//win)
#     x = windows.view(B, H//win, W//win, win, win, -1)
#     x = x.permute(0,1,3,2,4,5)
#     x = x.reshape(B,H,W, -1)
#     return x

# class WindowAttention(nn.Module):
#     def __init__(self, dim, num_heads, win):
#         super().__init__()

#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = self.dim//self.num_heads
#         self.scale = self.head_dim**-0.5
#         self.win = win

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)

#         self.proj = nn.Linear(dim, dim)
#         coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij"))
#         coords_flat = coords.flatten(1)
#         rel = coords_flat[:,:,None] - coords_flat[:,None, :]
#         rel = rel.permute(1,2,0)
#         rel[:,:,0] = rel[:,:,0] + [win - 1]
#         rel[:,:,1] = rel[:,:,0] + [win - 1]
#         rel[:,:,0] = rel[:,:,0] * [2*win - 1]
#         index = rel.sum(-1)
#         self.register_buffer = ("pos_index", index)
#         self.rel_bias = nn.Parameter(torch.zeroes((2*win-1) * (2*win-1), num_heads))

#     def forward(self, x , mask = None):
#         B_, N, C = x.shape
#         q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)

#         q = q * self.scale
#         attn = q @ k.transpose(-2,-1)
#         rb = self.rel_bias[self.pos_index.view(-1)].view(N,N,-1)
#         attn = attn + rb.permute(2,0,1).unsqueeze(0)

#         if mask is not None:
#             nw = mask.shape[0]
#             attn = attn.view(B_//nw, nw, self.num_heads, N, N)
#             attn = attn + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)

#         attn = attn.softmax(dim = -1)
#         out = (attn @ v).transpose(1,2).reshape(B_,N,C)
#         out = self.proj(out)
#         return out

# win = 3
# coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij"))
# coords_flat = coords.flatten(1)
# rel = coords_flat[:,:,None] - coords_flat[:,None, :]
# rel = rel.permute(1,2,0)

# rel[:,:,0] = rel[:,:,0] + [win - 1]
# rel[:,:,1] = rel[:,:,0] + [win - 1]
# rel[:,:,0] = rel[:,:,0] * [2*win - 1]
# index = rel.sum(-1)
# print(rel)


# class SwinBlock(nn.Module):
#     def __init__(self, dim, res, win, shift, heads):
#         super().__init__()
#         self.dim = dim
#         self.res = res
#         self.win = win
#         self.shift = shift

#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = WindowAttention(dim, heads, win)

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, 4*dim),
#             nn.GELU(),
#             nn.Linear(4*dim, dim)
#         )

#         H, W = res
#         if shift > 0:
#             self.mask = self.create_mask(H,W, win, shift)
#         else:
#             self.mask = None

#     def create_mask(self, H, W, win, shift):
#         img_mask = torch.zeros(1,H,W,1)
#         count = 0


#         for h in (slice(0, -win), (slice(-win, -shift)), slice(-shift, None)):
#             for w in ((slice(0, -win)), (slice(-win, -shift)), slice(-shift, None)):
#                 img_mask[:,h,w,:] = count
#                 count+=1

#         mask = window_partition(img_mask, win)
#         mask = mask.view(-1, win=win)
#         mask = mask.unsqueeze(1) - mask.unsqueeze(2)
#         mask = mask.masked_fill(mask != 0 - 10000.0)
#         return mask
    
#     def forward(self, x):
#         B,L,C = x.shape

#         H,W = self.res
#         residual = x
#         x = self.norm1(x)
#         x = x.view(B,H,W,C)

#         if self.shift > 0:
#             x = torch.roll(x, shifts = (-self.shift), dims=(1,2))

#         win_x = window_partition(x, self.win).view(-1, self.win*self.win, C)
#         attn_out = self.attn(win_x, self.mask.to(x.device) if self.mask is not None else None)

#         x = window_reverse(attn_out, self.win, H, W)

#         if self.shift > 0:
#             x = torch.roll(x, shifts=(+self.shift, +self.shift), dims=(1,2))

#         x = residual + x.view(B,L,C)

#         residual2 = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#         x = x + residual2

#         return x
    

# class PatchMerging(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#         self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
#         self.norm = nn.LayerNorm(4*dim)

#     def forward(self, x, H, W):
#         B,L,C = x.shape
#         x = x.view(B,H,W,C)
        
#         x0 = x[:, 0::2, 0::2, :]
#         x1 = x[:, 1::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, :]
#         x3 = x[:, 1::2, 1::2, :]

#         x0 = x0.reshape(B, -1, C)
#         x1 = x0.reshape(B, -1, C)
#         x2 = x0.reshape(B, -1, C)
#         x3 = x0.reshape(B, -1, C)

#         x = torch.cat([x0, x1, x2, x3], -1)

#         x = self.norm(1)
#         x = self.reduction(x)
#         return x, H//2, W//2
            
# class TwoStageSwinMNIST(nn.Module):
#     def __init__(self, embed_dim=48, heads=3, win=7, num_classes = 10):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)
#         initial_res = (14,14)

#         self.stage1_blocks = nn.Sequential(
#             SwinBlock(embed_dim, initial_res, heads=heads, win=win, shift=0),
#             SwinBlock(embed_dim, initial_res, heads=heads, win=win, shift=3)
#         )

#         # self.patch_merge = PatchMerging(embed_dim)
#         merged_dim = embed_dim

#         # stage2_res = (initial_res[0]//2, initial_res[1]//2)
#         # self.stage2_blocks = nn.Sequential(
#         #     SwinBlock(embed_dim, stage2_res, heads=heads, win=win, shift=0),
#         #     SwinBlock(embed_dim, stage2_res, heads=heads, win=win, shift=win//2)
#         # )

#         self.norm = nn.LayerNorm(merged_dim)
#         self.fc = nn.Linear(merged_dim, num_classes)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         B,C,H,W = x.shape
#         x = x.flatten(2).transpose(1,2)
#         L = H*W

#         x = self.stage1_blocks(x)
#         # X,H,W = self.patch_merge(x, H,W)
#         # L = H*W
#         # C= C*2

#         # x = self.stage2_blocks(x)

#         x = self.norm(x)
#         x = x.mean(dim = 1)
#         x = self.fc(x)

#         return x
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3801,))
# ])

# train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
# test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

# train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

# model = TwoStageSwinMNIST().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# loss_fn = nn.CrossEntropyLoss()


# def test(model):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for img, label in test_loader:
#             img = img.to(device)
#             label = label.to(device)
#             out = model(img)
#             pred = out.argmax(1)
#             correct += pred.eq(label).sum().item()
#             total += label.size(0)
#     return correct/total

# for epoch in range(5):
#     model.train()
#     for img, label in tqdm(train_loader):
#         img = img.to(device)
#         label = label.to(device)
#         out = model(img)
#         loss = loss_fn(out, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     acc = test(model)
#     print("Epoch", epoch + 1, "Accuracy =", acc*100)


# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# def window_partition(x, win):
#     B, H, W, C = x.shape
#     x = x.view(B, H // win, win, W // win, win, C)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
#     return x.view(-1, win, win, C)

# def window_reverse(windows, win, H, W):
#     Nw = windows.shape[0]
#     B = int(Nw // ((H // win) * (W // win)))
#     C = windows.shape[-1]
#     x = windows.view(B, H // win, W // win, win, win, C)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
#     return x.view(B, H, W, C)

# class WindowAttention(nn.Module):
#     def __init__(self, dim, num_heads, win):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.win = win
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)

#         coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij"))
#         coords_flat = coords.view(2, -1)
#         rel = coords_flat[:, :, None] - coords_flat[:, None, :]
#         rel = rel.permute(1, 2, 0)
#         rel[:, :, 0] += win - 1
#         rel[:, :, 1] += win - 1
#         rel[:, :, 0] *= 2 * win - 1
#         index = rel.sum(-1)
#         self.register_buffer("relative_index", index.long())
#         self.relative_bias = nn.Parameter(torch.zeros((2 * win - 1) ** 2, num_heads))

#     def forward(self, x, mask=None):
#         B_, N, C = x.shape
#         qkv = self.qkv(x).view(B_, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)

#         rb = self.relative_bias[self.relative_index.view(-1)]
#         rb = rb.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
#         attn = attn + rb

#         if mask is not None:
#             nw = mask.shape[0]
#             attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
#             attn = attn + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)

#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         return self.proj(x)

# class SwinBlock(nn.Module):
#     def __init__(self, dim, res, win, shift, heads):
#         super().__init__()
#         self.res = res
#         self.win = win
#         self.shift = shift
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = WindowAttention(dim, heads, win)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Linear(4 * dim, dim)
#         )

#         if shift > 0:
#             self.register_buffer("mask", self.create_mask(res[0], res[1], win, shift))
#         else:
#             self.register_buffer("mask", None)

#     def create_mask(self, H, W, win, shift):
#         img_mask = torch.zeros((1, H, W, 1))
#         count = 0
#         for h in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
#             for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
#                 img_mask[:, h, w, :] = count
#                 count += 1
#         mask = window_partition(img_mask, win).view(-1, win * win)
#         mask = mask.unsqueeze(1) - mask.unsqueeze(2)
#         return mask.masked_fill(mask != 0, -10000.0)

#     def forward(self, x):
#         B, L, C = x.shape
#         H, W = self.res
#         residual = x
#         x = self.norm1(x).view(B, H, W, C)

#         if self.shift > 0:
#             x = torch.roll(x, (-self.shift, -self.shift), (1, 2))

#         x = window_partition(x, self.win).view(-1, self.win * self.win, C)
#         x = self.attn(x, self.mask.to(x.device) if self.mask is not None else None)
#         x = window_reverse(x, self.win, H, W)

#         if self.shift > 0:
#             x = torch.roll(x, (self.shift, self.shift), (1, 2))

#         x = x.view(B, L, C) + residual
#         return x + self.mlp(self.norm2(x))

# class TwoStageSwinMNIST(nn.Module):
#     def __init__(self, embed_dim=48, heads=3, win=7):
#         super().__init__()
#         self.patch_embed = nn.Conv2d(1, embed_dim, 2, 2)
#         self.blocks = nn.Sequential(
#             SwinBlock(embed_dim, (14, 14), win, 0, heads),
#             SwinBlock(embed_dim, (14, 14), win, win // 2, heads)
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, 10)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.blocks(x)
#         x = self.norm(x).mean(1)
#         return self.fc(x)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
#     test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

#     train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

#     model = TwoStageSwinMNIST().to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#     loss_fn = nn.CrossEntropyLoss()

#     for epoch in range(5):
#         model.train()
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
#         for img, label in loop:
#             img, label = img.to(device), label.to(device)
#             loss = loss_fn(model(img), label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loop.set_postfix(loss=loss.item())

#         model.eval()
#         correct = 0
#         with torch.no_grad():
#             for img, label in test_loader:
#                 img, label = img.to(device), label.to(device)
#                 correct += (model(img).argmax(1) == label).sum().item()
#         print(f"Epoch {epoch+1} Accuracy = {100 * correct / len(test_data):.2f}%")



# if __name__ == "__main__":
#     main()


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
