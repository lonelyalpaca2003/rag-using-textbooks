from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from src.get_chroma_collection import create_or_load_vector_store
from src.load_docs import load_documents_with_metadata_included

def create_query_engine(db_name : str = 'ml_notes'):
    documents = load_documents_with_metadata_included()
    chroma_collection = create_or_load_vector_store(db_name)
    vector_store = ChromaVectorStore(chroma_collection= chroma_collection,)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context = storage_context)
    query_engine = index.as_query_engine(response_mode = 'tree_summarize', verbose = True, similarity_top_k = 10, 
                                         node_postprocessors = [CohereRerank(top_n = 5)])
    return query_engine



