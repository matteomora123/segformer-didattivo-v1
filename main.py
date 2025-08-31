# Mini Vision Transformer per Segmentazione Semantica (stile didattico)
# Dipendenze: torch, torchvision, pillow

import os, random
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ------------------ iperparametri ------------------
batch_size = 8
image_size = 256           # lato lungo dopo il resize
patch_size = 16            # patch quadrate P x P
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
n_classes = 21            # cambia secondo il tuo dataset
data_root = 'data'       # data/images/*.jpg  e  data/masks/*.png
seed = 1337
# ---------------------------------------------------

torch.manual_seed(seed)
random.seed(seed)

# ------------------ dataset ------------------
class SegFolderDataset(Dataset):
    """
    Si aspetta:
      root/images/*.jpg|png  immagini RGB
      root/masks/*.png       maschere con ID di classe per pixel (uint8)
    """
    def __init__(self, root, image_size, split='train', split_ratio=0.9):
        img_dir = os.path.join(root, 'images')
        msk_dir = os.path.join(root, 'masks')
        names = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.items = [(os.path.join(img_dir, n + os.path.splitext(f)[1]),
                       os.path.join(msk_dir, n + '.png'))
                      for n in names for f in [next(x for x in os.listdir(img_dir) if x.startswith(n))]]
        # split semplice e riproducibile
        random.Random(seed).shuffle(self.items)
        k = int(len(self.items)*split_ratio)
        self.items = self.items[:k] if split=='train' else self.items[k:]
        self.im_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.ms_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ipath, mpath = self.items[idx]
        img = Image.open(ipath).convert('RGB')
        msk = Image.open(mpath)         # deve contenere ID di classe
        img = self.im_tf(img)           # [3,H,W] float32 in [0,1]
        msk = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(self.ms_tf(msk).tobytes()))
            .numpy()
        )  # workaround per preservare indici; alternativa: torchvision.transforms.PILToTensor()
        msk = msk.view(image_size, image_size).long()  # [H,W] int64
        return img, msk

# loader
train_ds = SegFolderDataset(data_root, image_size, 'train')
val_ds   = SegFolderDataset(data_root, image_size, 'val')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

# ------------------ ViT componenti ------------------
class Head(nn.Module):
    """Una testa di self-attention (non causale)."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)               # (B,T,hs)
        q = self.query(x)             # (B,T,hs)
        v = self.value(x)             # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)  # (B,T,T)
        wei = F.softmax(wei, dim=-1)                         # (B,T,T)
        wei = self.dropout(wei)
        out = wei @ v                                         # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    """Encoder block ViT: LN->MHA+res, LN->MLP+res (pre-norm)."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ViTSegmenter(nn.Module):
    """
    Patch embedding -> +pos -> encoder transformer -> testa per classi per patch -> upsample a HxW.
    """
    def __init__(self, image_size, patch_size, n_classes):
        super().__init__()
        assert image_size % patch_size == 0
        self.Hp = self.Wp = image_size // patch_size
        self.n_patches = self.Hp * self.Wp
        self.patch_embed = nn.Conv2d(3, n_embd, kernel_size=patch_size, stride=patch_size)  # [B, C, Hp, Wp]
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

        # init semplice
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02);
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, imgs, targets=None):
        B, _, H, W = imgs.shape
        x = self.patch_embed(imgs)                  # (B, C, Hp, Wp)
        x = x.flatten(2).transpose(1,2)             # (B, T, C) con T = Hp*Wp
        x = x + self.pos_emb[:, :x.size(1), :]      # (B, T, C)
        x = self.blocks(x)                          # (B, T, C)
        x = self.ln_f(x)
        logits_patch = self.head(x)                 # (B, T, n_classes)
        logits = logits_patch.transpose(1,2).reshape(B, n_classes, self.Hp, self.Wp)
        logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)  # (B, n_classes, H, W)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)  # targets: (B,H,W) long
        return logits, loss

    @torch.no_grad()
    def predict(self, imgs):
        self.eval()
        logits, _ = self(imgs)
        return logits.argmax(1)   # (B,H,W) class ids

# ------------------ training & eval ------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    def run(loader, steps):
        losses = []
        for i, (x,y) in enumerate(loader):
            if i >= steps: break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        return sum(losses)/len(losses) if losses else float('nan')
    out = {'train': run(train_loader, eval_iters), 'val': run(val_loader, eval_iters)}
    model.train()
    return out

model = ViTSegmenter(image_size, patch_size, n_classes).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

step = 0
while step < max_iters:
    for xb, yb in train_loader:
        if step % eval_interval == 0 or step == max_iters-1:
            losses = estimate_loss(model)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step >= max_iters: break

# ------------------ inferenza esempio ------------------
# img_tensor shape: [1,3,H,W] in [0,1]; qui prendiamo un batch dalla val
model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred = model.predict(xb)[0].cpu().numpy()  # (H,W) int
        print("Pred shape:", pred.shape)
        break
