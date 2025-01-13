from setuptools import setup, find_packages

setup(
    name="mediminds",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'langchain',
        'langchain-community',
        'langchain-core',
        'langchain-huggingface',
        'streamlit',
        'python-dotenv',
        'faiss-cpu',
        'sentence-transformers',
        'PyPDF2',
        'python-magic'
    ]
)