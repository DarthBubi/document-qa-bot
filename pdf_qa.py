import os

import chromadb

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma


load_dotenv()


loader = PyPDFLoader(os.environ["DOCUMENT_PATH"])
documents = loader.load_and_split()

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="./chroma.db")

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.0), 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50}),
                                 return_source_documents=True)
