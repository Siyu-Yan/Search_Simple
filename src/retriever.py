# src/retriever.py
import os, json, argparse
import numpy as np
from embed import CLIPEncoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--emb_path", default="data/index.npy")
    ap.add_argument("--meta_path", default="data/index_meta.json")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    # 1) 载入向量库与元数据
    db = np.load(args.emb_path)  # (N, D)，已单位化
    with open(args.meta_path, "r", encoding="utf-8") as f:
        paths = json.load(f)["paths"]

    # 2) 编码查询（使用 embed.py 的默认模型：openai/clip-vit-base-patch32）
    enc = CLIPEncoder()  # 不传 model/device，避免不一致
    q = enc.encode_text(args.text).cpu().numpy().astype("float32")  # (D,)

    # 3) 余弦相似（单位化后用点积）
    scores = db @ q
    idx = np.argsort(-scores)[:args.top_k]

    results = [{"rank": r+1, "path": paths[i], "score": float(scores[i])}
               for r, i in enumerate(idx)]
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
