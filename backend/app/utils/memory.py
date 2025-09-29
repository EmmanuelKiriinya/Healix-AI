# app/utils/memory.py
from langchain.vectorstores import FAISS

def save_faiss(index: FAISS, path: str):
    index.save_local(path)

def load_faiss(path: str, embeddings):
    return FAISS.load_local(path, embeddings)
