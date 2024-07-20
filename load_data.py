from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_data(file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    loader = PyPDFLoader(file)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    return docs


def embed_data(docs):
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    database = Chroma.from_documents(docs, embeddings, persist_directory="chromadb")
    return database

