from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import textwrap
import pandas as pd
import re

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_data(file_path=None, df=None):
    if df is None:
        df = pd.read_excel(file_path)
    
    required_columns = [
        'title', 'address', 'property_id', 'bedroom', 
        'bathroom', 'property_type', 'property_status', 
        'guest number', 'city', 'area'
    ]
    
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    text_columns = ['title', 'address', 'property_type', 'property_status', 'city', 'area']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(preprocess_text)
    
    df['combined_text'] = df.apply(lambda row: ' '.join(
        str(row[col]) for col in available_columns if col in row
    ), axis=1)
    
    documents = []
    for _, row in df.iterrows():
        metadata = {
            'property_id': str(row.get('property_id', '')),
            'bedroom': str(row.get('bedroom', '')),
            'bathroom': str(row.get('bathroom', '')),
            'guest_number': str(row.get('guest number', ''))
        }
        
        for field in text_columns:
            if field in row:
                metadata[field] = row[field]
        
        doc = Document(
            page_content=row['combined_text'],
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents=documents)

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': normalize_embedding}
    )

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def get_response(query, chain):
    response = chain({'query': query})
    return response