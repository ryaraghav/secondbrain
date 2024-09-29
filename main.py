import json
from dotenv import load_dotenv
import os
from src.loaders import load_youtube_data, load_text_data
from src.tools import create_retriever
from src.agent import create_agent
from langchain_openai import ChatOpenAI

# Load LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load config file
with open('config/sources.json', 'r') as f:
    sources = json.load(f)
    print(sources)

# Create tools (retrievers)
tools = []
for source in sources:
    if source['type'] == 'youtube':
        data = load_youtube_data(source['path'])
    elif source['type'] == 'text':
        data = load_text_data(source['path'])
    else:
        raise ValueError(f"Unsupported type: {source['type']}")

    retriever = create_retriever(data, source['tool_name'], source['description'])
    tools.append(retriever)

# Load prompt
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create agent and execute
agent_executor = create_agent(llm, tools, prompt)
response = agent_executor.invoke({"input": "how can I prepare to delegate a task"})
print(response)