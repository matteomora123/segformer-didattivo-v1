# train.py  —  (tua versione, modifiche minime)
import os, random
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import ImageOps, ImageFilter
import numpy as np, glob
# ------------------ iperparametri ------------------
batch_size = 8
image_size = 256
patch_size = 8
max_iters = 1500
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
n_classes = 5
data_root = '.'
seed = 1337
weights_path = 'vit_seg.pt'  # <-- NEW
# ---------------------------------------------------

torch.manual_seed(seed)
random.seed(seed)

def compute_class_weights(msk_dir, n_classes, device):
    counts = np.zeros(n_classes, dtype=np.int64)
    for p in glob.glob(os.path.join(msk_dir, '*.png')):
        a = np.array(Image.open(p))
        vals, cnts = np.unique(a, return_counts=True)
        for v, c in zip(vals, cnts):
            if 0 <= v < n_classes:
                counts[v] += c
    freqs = counts / counts.sum()
    nz = freqs[freqs > 0]
    med = np.median(nz) if len(nz) else 1.0
    w = np.zeros_like(freqs, dtype=np.float32)
    w[freqs > 0] = med / freqs[freqs > 0]      # “median frequency balancing”
    # opzionale: limita gli estremi
    w = np.clip(w, 0.5, 5.0)
    return torch.tensor(w, device=device)

CLASS_W = compute_class_weights(os.path.join(data_root, 'masks'), n_classes, device)
print("Class weights:", CLASS_W.tolist())

def _prep(img):
    g = img.convert('L')
    g = ImageOps.autocontrast(g)
    g = ImageOps.invert(ImageOps.invert(g).filter(ImageFilter.MaxFilter(3)))
    return g.convert('RGB')

def _is_img(fname):
    f = fname.lower()
    return f.endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))

class SegFolderDataset(Dataset):
    def __init__(self, root, image_size, split='train', split_ratio=0.9):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.img_dir = os.path.join(root, 'images')
        self.msk_dir = os.path.join(root, 'masks')
        if not os.path.isdir(self.img_dir): raise FileNotFoundError(f"images/ non trovata: {self.img_dir}")
        if not os.path.isdir(self.msk_dir): raise FileNotFoundError(f"masks/ non trovata: {self.msk_dir}")
        imgs = sorted([f for f in os.listdir(self.img_dir) if _is_img(f)])
        pairs = []
        for f in imgs:
            base, _ = os.path.splitext(f)
            mpath = os.path.join(self.msk_dir, base + '.png')
            ipath = os.path.join(self.img_dir, f)
            if os.path.isfile(mpath):
                pairs.append((ipath, mpath))
        if not pairs: raise RuntimeError("Nessuna coppia immagine/maschera trovata.")
        random.Random(seed).shuffle(pairs)
        k = int(len(pairs)*split_ratio)
        self.items = pairs[:k] if split=='train' else pairs[k:]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        import numpy as np
        ipath, mpath = self.items[idx]
        img = _prep(Image.open(ipath))
        msk = Image.open(mpath).convert('L')
        img = img.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
        msk = msk.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        arr_img = np.array(img, copy=True)    # evita warning “non writable”
        arr_msk = np.array(msk, copy=True)
        img_t = torch.from_numpy(arr_img).permute(2,0,1).float() / 255.0
        mask_t = torch.from_numpy(arr_msk).long()
        return img_t, mask_t

def make_loaders():
    train_ds = SegFolderDataset(data_root, image_size, 'train')
    val_ds   = SegFolderDataset(data_root, image_size, 'val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0, pin_memory=(device=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device=='cuda'))
    return train_loader, val_loader

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        wei = (q @ k.transpose(-2,-1)) * (k.shape[-1]**-0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
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
    def __init__(self, image_size, patch_size, n_classes):
        super().__init__()
        assert image_size % patch_size == 0
        self.Hp = self.Wp = image_size // patch_size
        self.n_patches = self.Hp * self.Wp
        self.patch_embed = nn.Conv2d(3, n_embd, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_classes)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, imgs, targets=None):
        B, _, H, W = imgs.shape
        x = self.patch_embed(imgs)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits_patch = self.lm_head(x)
        logits = logits_patch.transpose(1,2).reshape(B, n_classes, self.Hp, self.Wp)
        logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, targets, weight=CLASS_W) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def predict(self, imgs):
        self.eval()
        logits, _ = self(imgs)
        return logits.argmax(1)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
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

def run():
    train_loader, val_loader = make_loaders()
    model = ViTSegmenter(image_size, patch_size, n_classes).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    step = 0
    while step < max_iters:
        for xb, yb in train_loader:
            if step % eval_interval == 0 or step == max_iters-1:
                losses = estimate_loss(model, train_loader, val_loader)
                print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= max_iters: break
    torch.save(model.state_dict(), weights_path)  # <-- NEW
    print(f"Salvato: {weights_path}")            # <-- NEW
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model.predict(xb)[0].cpu().numpy()
            print("Pred shape:", pred.shape)
            break

if __name__ == "__main__":
    run()
