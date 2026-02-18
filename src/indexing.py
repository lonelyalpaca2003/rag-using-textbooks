from dotenv import load_dotenv
import os
from llama_index.core import Document
import pypdf
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank

load_dotenv()

def load_documents_with_metadata_included(data_path:str = 'data'):
    all_docs = []
    for filename in os.listdir(data_path):
        if not filename.endswith('.pdf'):
            continue
        file_path = os.path.join(data_path, filename)
        reader = pypdf.PdfReader(file_path)

        if "lecture" in filename.lower():
            doc_type = "lecture"

        if 'lecture' not in filename.lower():
            doc_type = "textbook"

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue
            doc = Document(text = text, 
                           metadata = {
                               "file_name" : filename, 
                               "page_num" : page_num, 
                               "doc_type" : doc_type,
                               "course" : "Machine Learning"
                           })     
            all_docs.append(doc)
            

    return all_docs 

def create_or_load_vector_store(db_name: str = "ml_notes"):
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    Settings.embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    Settings.node_parser = splitter
    chroma_client = chromadb.PersistentClient(path='./chroma')
    
    try:
        chroma_collection = chroma_client.get_collection(db_name)   
        print(f"Loading existing database {db_name}") 
        
        if chroma_collection.count() == 0:  
            print("Collection is empty! Re-indexing...")
            chroma_client.delete_collection(db_name)
            chroma_collection = chroma_client.create_collection(db_name)  
        else:
            print(f"Collection has {chroma_collection.count()} items") 
            
    except ValueError:     
        chroma_collection = chroma_client.create_collection(db_name) 
        print(f"Creating new database {db_name}")

    return chroma_collection


def create_query_engine(db_name : str = 'ml_notes'):
    documents = load_documents_with_metadata_included()
    chroma_collection = create_or_load_vector_store(db_name)
    vector_store = ChromaVectorStore(chroma_collection= chroma_collection,)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context = storage_context)
    query_engine = index.as_query_engine(response_mode = 'tree_summarize', verbose = True, similarity_top_k = 10, 
                                         node_postprocessors = [CohereRerank(top_n = 5)])
    return query_engine, index
