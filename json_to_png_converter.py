# convert_labelme_to_masks.py
import os, json, glob
import numpy as np
from PIL import Image, ImageDraw

# mappa classi -> ID (0 = sfondo). Cambia se servono altri nomi.
CLASS2ID = {"star":1, "square":2, "circle":3, "triangle":4}

ROOT = r"C:\Users\matte\PycharmProjects\segformer-didattivo-v1"
ANN_DIR = os.path.join(ROOT, "annotated")
OUT_DIR = os.path.join(ROOT, "masks")
os.makedirs(OUT_DIR, exist_ok=True)

def convert_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # recupera il nome dell'immagine
    img_name = data.get("imagePath")
    if img_name is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        # supponiamo sia .jpg con stesso nome del json
        img_name = base + ".jpg"
    img_path = os.path.join(os.path.dirname(json_path), "..", "images", img_name)
    img_path = os.path.abspath(img_path)

    # apri l'immagine per avere width e height
    with Image.open(img_path) as im:
        w, h = im.size

    # maschera vuota
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for sh in data.get("shapes", []):
        cls = CLASS2ID.get(sh.get("label"))
        if cls is None:
            continue
        pts = [(int(x), int(y)) for x, y in sh["points"]]
        draw.polygon(pts, outline=cls, fill=cls)

    base = os.path.splitext(os.path.basename(img_name))[0]
    out_path = os.path.join(OUT_DIR, f"{base}.png")
    mask.save(out_path)

def main():
    files = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    for p in files: convert_one(p)
    print(f"Fatti {len(files)} file. Maschere in {OUT_DIR}")

if __name__ == "__main__":
    main()
