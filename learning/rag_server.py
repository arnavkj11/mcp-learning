import os
import logging
from mcp.server.fastmcp import FastMCP
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb import Settings

CHROMA_PERSIST_DIR = "rag_chroma_db"

mcp = FastMCP("RAGAssistant")

mcp.tool()
def ingest_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"Error: File {file_path} does not exist."
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            return "Error: No text chunks were created from the document."
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=Settings(anonymized_telemetry=False)
        )

        file_name = os.path.basename(file_path)
        return f"Document '{file_name}' ingested successfully with {len(chunks)} chunks."
    
    except Exception as e:
        print(f"Error ingesting document: {e}")
        return f"Error ingesting document: {e}"
    
if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    print("Starting RAG MCP Server...")
    mcp.run(transport="stdio")
    