import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

class LLMConfig:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.db_path = "vector_db/"
        
    def load_llm(self):
        """Initialize and return the LLM"""
        llm = HuggingFaceEndpoint(
            repo_id=self.huggingface_repo_id,
            temperature=0.5,
            model_kwargs={
                "max_length": 512,
                "token": self.hf_token
            }
        )
        return llm

    def get_prompt_template(self):
        """Return the custom prompt template"""
        template = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def load_vectorstore(self):
        """Load the FAISS vectorstore"""
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.load_local(
            self.db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )

    def create_qa_chain(self):
        """Create and return the QA chain"""
        db = self.load_vectorstore()
        return RetrievalQA.from_chain_type(
            llm=self.load_llm(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.get_prompt_template()}
        )