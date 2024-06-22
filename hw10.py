# 參考 https://github.com/LeeYi-user/ai/blob/master/homework/10/react.py
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

os.environ["TAVILY_API_KEY"] = ""

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("hwchase17/react")

llm = ChatGroq(api_key="")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input_data = {
    "input": "現在幾點? 用中文回答"
}

response = agent_executor.invoke(input_data)

print("Response:", response)