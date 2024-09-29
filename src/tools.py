from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

import os
load_dotenv(dotenv_path="/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_retriever(data, tool_name, description):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(data)
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory="data/chroma_db")
    retriever = vectordb.as_retriever()
    return create_retriever_tool(retriever, tool_name, description)

#Testing - DELETE THIS
from langchain_community.document_loaders import YoutubeLoader
data = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=pnrpPWGDyyA", add_video_info=False).load()
retriever = create_retriever(data, "test", "for motivation")
