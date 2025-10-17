## Minimal CLIP Retrieval (Baseline)
1) 安装依赖
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

2) 下载与解压数据
   python3 -m pip install "gdown[all]"
   python3 -m gdown --id 1Sh_CC0h-idHnAdQXh3V9uTVyGVGqOHj8
   unzip small_dataset.zip -d data/

3) 建索引
   python3 src/index.py --images_dir data/small_dataset

4) 查询
   python3 src/retriever.py --text "Click on the 'Submit' button" --top_k 5