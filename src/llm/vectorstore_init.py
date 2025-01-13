import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

class VectorStoreInitializer:
    def __init__(self):
        self.data_path = os.path.join(project_root, "data/documents/")
        self.db_path = os.path.join(project_root, "vector_db/")
        
    def load_documents(self):
        """Load PDF documents from the data directory"""
        loader = DirectoryLoader(
            self.data_path, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader
        )
        return loader.load()

    def create_text_chunks(self, documents):
        """Split documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        return splitter.split_documents(documents)

    def get_embedding_model(self):
        """Initialize and return the embedding model"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def initialize_vectorstore(self):
        """Create and save the FAISS vectorstore"""
        documents = self.load_documents()
        text_chunks = self.create_text_chunks(documents)
        embedding_model = self.get_embedding_model()
        
        # Create and save vectorstore
        db = FAISS.from_documents(text_chunks, embedding_model)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        db.save_local(self.db_path)
        print(f"Vectorstore created and saved at {self.db_path}")

if __name__ == "__main__":
    vectorstore_init = VectorStoreInitializer()
    vectorstore_init.initialize_vectorstore()