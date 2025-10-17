
#!/bin/bash
set -e  # 出错时立刻退出

# === Configuration ===
VENV_PATH=".venv/bin/activate"
IMAGES_DIR="data/small_dataset"
EMB_PATH="data/index.npy"
META_PATH="data/index_meta.json"
TOP_K=5

# === Step 0. 激活虚拟环境 ===
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found. Please run: python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

# === Step 1. 检查索引是否存在 ===
if [ ! -f "$EMB_PATH" ] || [ ! -f "$META_PATH" ]; then
    echo "⚙️  No index found — building index..."
    python3 src/index.py --images_dir "$IMAGES_DIR" --emb_path "$EMB_PATH" --meta_path "$META_PATH"
else
    echo "✅ Found existing index — skipping build."
fi

# === Step 2. 检索文本输入 ===
QUERY="${1:-Click on the 'Continue' button}"

echo "🔍 Running retrieval for query: \"$QUERY\""
python3 src/retriever.py \
    --text "$QUERY" \
    --emb_path "$EMB_PATH" \
    --meta_path "$META_PATH" \
    --top_k $TOP_K
