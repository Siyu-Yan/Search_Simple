webui-retrieval/
  requirements.txt
  README.md
  data/                  # 放图片数据集 (small_dataset 解压到这里)
  src/
    embed.py             # CLIP 封装
    index.py             # 建索引：图片→向量，并保存
    retriever.py         # 查询：文本→向量，做 Top-K 检索

main function contains:
 1. embed image 
 2. embed query text
 3. retrieve top k 
 
