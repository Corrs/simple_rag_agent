"""
agent.py
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from rag_agent.retriever_tool import retrieve_context
from utils.env_util import ZAI_API_KEY, ZAI_BASE_URL

llm = ChatOpenAI(
  temperature=0.6,
  model="glm-4.5",
  api_key=ZAI_API_KEY,
  base_url=ZAI_BASE_URL,
  stream_usage=True
)

agent = create_agent(
  model=llm,
  tools=[retrieve_context],
  system_prompt="""
  你是一个Java面试问答助手。请结合工具回答用户提问的Java面试问题，如果用户的问题超出了Java面试的范围，请回答你不知道！
  - retrieve_context Java面试问题检索工具
  如果工具检索的答案和用户问题不相关，请回答未检索到该问题的答案！
  """
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "Java中如何使用线程池"}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

for event in agent.stream(
    {"messages": [{"role": "user", "content": "潍坊市在哪里？"}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()