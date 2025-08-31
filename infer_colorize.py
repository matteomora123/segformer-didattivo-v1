# infer_colorize.py — carica vit_seg.pt e salva *_result.png in "sample test"
import os, sys, glob, numpy as np
import torch, torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageOps, ImageFilter
# ---- iperparametri (DEVONO combaciare col training) ----
image_size = 256
patch_size = 8  # <--- metti qui lo stesso valore usato in train.py (es. 8 se hai riaddestrato)
n_embd, n_head, n_layer, dropout = 384, 6, 6, 0.1
n_classes = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_path = 'vit_seg.pt'
sample_dir = os.path.join('.', 'sample test')

# colori: 1=stella blu, 2=quadrato rosso, 3=cerchio giallo, 4=triangolo verde
PALETTE = {1:(0,0,255), 2:(255,0,0), 3:(255,255,0), 4:(0,255,0)}

def _prep(img):
    g = img.convert('L')                      # scala di grigi
    g = ImageOps.autocontrast(g)              # alza il contrasto
    g = ImageOps.invert(ImageOps.invert(g).filter(ImageFilter.MaxFilter(3)))  # dilata linee sottili
    return g.convert('RGB')                   # torna a 3 canali identici

# ---- architettura identica al training (stessi nomi attributi) ----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        k,q,v = self.key(x), self.query(x), self.value(x)
        wei = (q @ k.transpose(-2,-1)) * (k.shape[-1]**-0.5)
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))

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
    def forward(self, imgs):
        B,_,H,W = imgs.shape
        x = self.patch_embed(imgs)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits_patch = self.lm_head(x)
        logits = logits_patch.transpose(1,2).reshape(B, n_classes, self.Hp, self.Wp)
        logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
        return logits
# -----------------------------------------------------

def colorize(mask_L):
    arr = np.array(mask_L, copy=True); H,W = arr.shape
    out = np.full((H,W,3), 255, np.uint8)
    for cid, col in PALETTE.items(): out[arr==cid] = col
    return Image.fromarray(out, 'RGB')

def next_output_name(in_path):
    folder = os.path.dirname(in_path); base,_ = os.path.splitext(os.path.basename(in_path))
    out = os.path.join(folder, f"{base}_result.png"); i = 1
    while os.path.exists(out):
        out = os.path.join(folder, f"{base}_result({i}).png"); i += 1
    return out

@torch.no_grad()
def infer_file(model, img_path):
    # grayscale->RGB per ridurre domain shift (3 canali identici)
    img = _prep(Image.open(img_path))
    W0, H0 = img.size
    arr = np.array(img, copy=True)
    x = torch.from_numpy(arr).permute(2,0,1).float()/255.0
    x = F.interpolate(x.unsqueeze(0), size=(image_size,image_size),
                      mode='bilinear', align_corners=False).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)  # [1, nC, H, W]
    conf, pred = probs.max(1)  # [1, H, W] ciascuno
    pred = pred[0].cpu().numpy().astype(np.uint8)
    conf = conf[0].cpu().numpy()

    TAU = 0.60  # soglia: abbassa o alza se serve (0.5–0.7 tipico)
    pred[conf < TAU] = 0  # forza background quando la conf è bassa

    # riportiamo la maschera alla risoluzione originale
    pred = np.array(Image.fromarray(pred, 'L').resize((W0, H0), Image.Resampling.NEAREST))

    vals, cnts = np.unique(pred, return_counts=True)
    print("classi_pred:", dict(zip(vals.tolist(), cnts.tolist())))

    base, _ = os.path.splitext(img_path)
    mask_path = next_output_name(base + "_maskIDs.png")
    Image.fromarray(pred, 'L').save(mask_path)

    color_img = colorize(Image.fromarray(pred, 'L'))
    out_path = next_output_name(img_path)
    color_img.save(out_path)

    # overlay sull'ORIGINALE (non preprocessato)
    arr_orig = np.array(Image.open(img_path).convert('RGB'))
    overlay = arr_orig.copy()
    for cid, col in PALETTE.items():
        m = pred == cid
        overlay[m] = (0.6 * overlay[m] + 0.4 * np.array(col, np.uint8)).astype(np.uint8)
    overlay_path = next_output_name(base + "_overlay.png")
    Image.fromarray(overlay, 'RGB').save(overlay_path)

    print("Salvati:", mask_path, "|", out_path, "|", overlay_path)

def main():
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Pesi non trovati: {weights_path} (esegui prima: python train.py)")
    state = torch.load(weights_path, map_location=device)  # state_dict
    model = ViTSegmenter(image_size, patch_size, n_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    if len(sys.argv) > 1:
        infer_file(model, sys.argv[1])
    else:
        os.makedirs(sample_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
        if not files:
            print(f"Nessun PNG in: {sample_dir}"); return
        for p in files: infer_file(model, p)

if __name__ == '__main__':
    main()
