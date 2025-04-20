# Importing the necessary packages
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import textwrap
import pandas as pd
import re

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Load data from Excel (file path or DataFrame)
def load_data(file_path=None, df=None):
    """
    Load data from Excel file or DataFrame with specific property columns
    """
    if df is None:
        df = pd.read_excel(file_path)
    
    # Select required columns
    required_columns = [
        'title', 'address', 'property_id', 'bedroom', 
        'bathroom', 'property_type', 'property_status', 
        'guest number', 'city', 'area'
    ]
    
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Text preprocessing
    text_columns = ['title', 'address', 'property_type', 'property_status', 'city', 'area']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(preprocess_text)
    
    # Combine all text for embedding
    df['combined_text'] = df.apply(lambda row: ' '.join(
        str(row[col]) for col in available_columns if col in row
    ), axis=1)
    
    # Convert to LangChain documents
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

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

# Load embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding
        }
    )

# Create embeddings
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

# QA Chain
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Get formatted response
def get_response(query, chain):
    response = chain({'query': query})
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)

# Templates
prompt = """
### System:
You are an AI Assistant that follows instructions extremely well. \
Help as much as you can.

### User:
{prompt}

### Response:
"""

template = """
### System:
You are a respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""