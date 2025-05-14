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
        self.huggingface_repo_id = "HuggingFaceH4/zephyr-7b-beta"
        self.db_path = "vector_db/"

    def load_llm(self):
        """Initialize and return the LLM"""
        llm = HuggingFaceEndpoint(
            repo_id=self.huggingface_repo_id,
            huggingfacehub_api_token=self.hf_token,
            task="text-generation",  # Zephyr supports text-generation
            temperature=0.5,
            max_new_tokens=512,
            top_p=0.95,
            do_sample=True
        )
        return llm

    def get_prompt_template(self):
        """Return the custom prompt template"""
        template = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context.

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