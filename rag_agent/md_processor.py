"""
markdown文档处理工具
"""
import os
from typing import List

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 获取当前文件目录
file_path = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_path, "files")

# embedding模型
faiss_dir = "faiss"
index_name = "java_qa_index"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


def load_documents() -> List[Document]:
  """加载markdown文档"""

  # 获取目录下的文件
  file_names = os.listdir(file_dir)
  docs = []
  for file_name in file_names:
    file = os.path.join(file_dir, file_name)
    loader = UnstructuredAPIFileLoader(
      file_path=file,
      mode="elements",
      strategy="hi_res",
      url="http://192.168.0.104:8888/general/v0/general",
      coordinates=True,
      encoding="utf-8",
      extract_image_block_types=["Image", "Table"],
      hi_res_model_name="yolox"
    )
    for doc in loader.lazy_load():
      docs.append(doc)

  return docs


def split_documents(docs: List[Document]) -> List[Document]:
  """将markdown文档进行分段，将文档中超过1000字符的进行切割"""
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个chunk最大字符数
    chunk_overlap=200,  # chunk间重叠字符数，保持上下文
    add_start_index=True,  # 添加原始位置索引
    separators=[
      "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""
    ],  # 中文分割符优化
    keep_separator=True,  # 保留分隔符
  )
  return text_splitter.split_documents(docs)


def store_documents(docs: List[Document]):
  """将分段后的文档进行存储"""
  index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))
  vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
  )
  vector_store.add_documents(documents=docs)
  # 保存到本地
  vector_store.save_local(faiss_dir, index_name=index_name)


retriever = FAISS.load_local(
  faiss_dir,
  embeddings,
  index_name=index_name,
  allow_dangerous_deserialization=True
).as_retriever(search_type="similarity", search_kwargs={"k": 3})

if __name__ == '__main__':
  store_documents(split_documents(load_documents()))
