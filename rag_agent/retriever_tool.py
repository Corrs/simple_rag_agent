"""
retriever_tool.py
数据检索工具
"""
from langchain_core.tools import tool

from rag_agent.md_processor import retriever


@tool(response_format="content_and_artifact", parse_docstring=True)
def retrieve_context(query: str):
  """数据检索工具，从数据库中检索问题对应的答案

  Args:
    query: str 问题

  Return:
    tuple，返回一个包含(content, artifact)的元组
  """
  # 检索数据
  retrieved_docs = retriever.invoke(query)

  serialized = "\n\n".join(
    f"Source: {doc.metadata}\nContent: {doc.page_content}"
    for doc in retrieved_docs
  )

  return serialized, retrieved_docs