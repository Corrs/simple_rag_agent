from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
  model_name="BAAI/bge-small-zh-v1.5"
)