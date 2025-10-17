# src/index.py  —— 极简版（贴近你的 Colab 原型）
import os, glob, json, argparse
import numpy as np
from PIL import Image
from embed import CLIPEncoder  # 我们已用 transformers 封装好了

def list_images(root, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser(description="Build a minimal CLIP index (NumPy).")
    ap.add_argument("--images_dir", required=True, help="Folder containing screenshots")
    ap.add_argument("--emb_path", default="data/index.npy")
    ap.add_argument("--meta_path", default="data/index_meta.json")
    args = ap.parse_args()

    # 1) 找图
    paths = list_images(args.images_dir)
    if not paths:
        raise SystemExit(f"No images found under: {args.images_dir}")

    # 2) 加载图片（保持简单，不做额外预处理）
    imgs = [Image.open(p).convert("RGB") for p in paths]

    # 3) 编码（transformers-CLIP，已在 embed.py 封装了）
    enc = CLIPEncoder()  # 默认: model="openai/clip-vit-base-patch32", 自动选 cuda/cpu
    feats = enc.encode_images(imgs).cpu().numpy().astype("float32")  # (N, D)，已 L2 归一化

    # 4) 保存
    os.makedirs(os.path.dirname(args.emb_path), exist_ok=True)
    np.save(args.emb_path, feats)
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump({"paths": paths}, f, indent=2)

    print(f"✅ Saved vectors: {args.emb_path}  shape={feats.shape}")
    print(f"✅ Saved metadata: {args.meta_path} (n={len(paths)})")

if __name__ == "__main__":
    main()
