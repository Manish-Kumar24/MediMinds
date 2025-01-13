import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration class for the application"""
    HF_TOKEN: str
    REPO_ID: str
    MODEL_KWARGS: dict
    VECTOR_DB_PATH: str
    
    @classmethod
    def load_config(cls):
        """Load configuration from environment variables"""
        load_dotenv()
        return cls(
            HF_TOKEN=os.getenv("HF_TOKEN"),
            REPO_ID="mistralai/Mistral-7B-Instruct-v0.3",
            MODEL_KWARGS={
                "max_length": 512,
                "temperature": 0.5
            },
            VECTOR_DB_PATH="vector_db/index.faiss"
        )