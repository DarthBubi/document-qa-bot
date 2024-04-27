import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentStore:

    def __init__(self, path: str = None) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.documents = None
        self.docsearch = None
        self.path: str = path if path is not None else "./chromadb"

    def generate_embeddings(self) -> None:
        path = os.environ["DOCUMENT_PATH"]
        loader = PyPDFDirectoryLoader(path, glob="*5D*.pdf")
        self.documents = loader.load_and_split()

    def get_embeddings(self) -> OpenAIEmbeddings:
        return self.embeddings

    def fill_chroma(self) -> None:
        if not os.path.exists(self.path):
            self.docsearch = Chroma.from_documents(self.documents, self.embeddings, persist_directory=self.path)
        else:
            self.docsearch = Chroma(persist_directory=self.path, embedding_function=self.embeddings)

    def get_chroma(self) -> Chroma:
        if self.docsearch is None:
            self.docsearch = Chroma(persist_directory=self.path, embedding_function=self.embeddings)
        return self.docsearch
    