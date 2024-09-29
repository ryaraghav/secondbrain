from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor

def create_agent(llm, tools, prompt):
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=False, verbose = False)
    return agent_executor