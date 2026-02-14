from dotenv import load_dotenv
import os
from llama_index.core import Document
import pypdf

def load_documents_with_metadata_included(data_path:str = '../data'):
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