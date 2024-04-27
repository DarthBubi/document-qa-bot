import os

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from tools.document_store import DocumentStore

import gradio as gr

load_dotenv()

model = os.environ["MODEL_NAME"]
path = os.environ["DOCUMENT_PATH"]
chromadb_path = os.environ["CHROMADB_PATH"]

if not os.path.exists(chromadb_path):
    docsearch = DocumentStore(path=chromadb_path)
    docsearch.generate_embeddings()
    docsearch.fill_chroma()
else:
    docsearch = DocumentStore(path=chromadb_path)

llm = ChatOpenAI(model_name=model, temperature=0.0, max_tokens=2048)
retriever = docsearch.get_chroma().as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50})
prompt_template = PromptTemplate.from_template(os.environ["PROMPT_TEMPLATE"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

set_llm_cache(InMemoryCache())

def qa_endpoint(message, history):
    if message is not None:      
        response = rag_chain.invoke(message)
        # source_documents = [doc.metadata['source'] + ' p.'  + str(doc.metadata['page']) for doc in response['source_documents']]

        return response
    
def qa_endpoint_streaming(message, history):
    if message is not None:
        prompt_template = PromptTemplate.from_template(os.environ["PROMPT_TEMPLATE"])
        query = prompt_template.format(query=message)
        partial_message = ""
        for chunk in rag_chain.stream({'query': query}):
            partial_message = partial_message + chunk['result']
            yield partial_message
            if 'source_documents' in chunk:
                source_documents = [doc.metadata['source'] + ' p.'  + str(doc.metadata['page']) + '\n'  for doc in chunk['source_documents']]
                yield partial_message + "\n" + "\n".join(source_documents)

gr.ChatInterface(qa_endpoint).queue().launch(share=False, debug=True)
