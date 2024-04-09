import os
import glob

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.globals import set_llm_cache
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

import gradio as gr

load_dotenv()

embeddings = OpenAIEmbeddings()
model = os.environ["MODEL_NAME"]
path = os.environ["DOCUMENT_PATH"]

if not os.path.exists("./chromadb"):
    loader = PyPDFDirectoryLoader(path, glob="*5D*.pdf")
    documents = loader.load_and_split()

    docsearch = Chroma.from_documents(documents, embeddings, persist_directory="./chromadb")
else:
    docsearch = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name=model, temperature=0.0, max_tokens=2048),
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50}),
                                    return_source_documents=True)
set_llm_cache(InMemoryCache())

def qa_endpoint(message, history):
    if message is not None:
        prompt_template = PromptTemplate.from_template(os.environ["PROMPT_TEMPLATE"])
        query = prompt_template.format(query=message)
        response = qa(query)
        source_documents = [doc.metadata['source'] + ' p.'  + str(doc.metadata['page']) for doc in response['source_documents']]

        return response["result"].join(source_documents)
    
def qa_endpoint_streaming(message, history):
    if message is not None:
        prompt_template = PromptTemplate.from_template(os.environ["PROMPT_TEMPLATE"])
        query = prompt_template.format(query=message)
        partial_message = ""
        for chunk in qa.stream({'query': query}):
            partial_message = partial_message + chunk['result']
            yield partial_message
            if 'source_documents' in chunk:
                source_documents = [doc.metadata['source'] + ' p.'  + str(doc.metadata['page']) + '\n'  for doc in chunk['source_documents']]
                yield partial_message + "\n" + "\n".join(source_documents)

gr.ChatInterface(qa_endpoint_streaming).queue().launch(share=True, debug=True)
