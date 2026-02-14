import os
import chromadb
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def create_or_load_vector_store(db_name : str = "ml_notes"):
    splitter = SentenceSplitter(chunk_size = 512, chunk_overlap= 100)
    Settings.embed_model = HuggingFaceEmbedding(model_name = 'sentence-transformers/all-MiniLM-L6-V2')
    Settings.node_parser = splitter
    chroma_client = chromadb.PersistentClient(path = './chroma')
    try:
        ## load directly if the database exists 
        chroma_collection = chroma_client.get_collection(db_name)   
        print(f"Loading existing database {db_name}") 
    except:     
        ## if the database doesn't exist make a new collection 
        chroma_collection = chroma_client.create_collection(db_name) 
        print(f"Creating new database {db_name} ")

    return chroma_collection


        