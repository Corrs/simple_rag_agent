import os

from dotenv import load_dotenv

load_dotenv(override=True)

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL")
ZAI_SEARCH_API_KEY = os.getenv("ZAI_SEARCH_API_KEY")

MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_URL = os.getenv("MILVUS_URL")
MILVUS_DB = os.getenv("MILVUS_DB")
