from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
import pandas as pd
import re
from langdetect import detect

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return text

def detect_query_language(query):
    try:
        return detect(query)
    except:
        return 'en'  # default to English if detection fails

def format_property_info(row, language='en'):
    title = row.get('title', '')
    property_type = row.get('property_type', '')
    address = row.get('address', '')
    city = row.get('city', '')
    area = row.get('area', '')
    bedroom = row.get('bedroom', '')
    bathroom = row.get('bathroom', '')
    guest_number = row.get('guest_number', '')
    price_info = str(row.get('price_info', ''))
    
    # Format property type with quotes
    property_type_str = f"'{property_type}'" if property_type else ''
    
    if language == 'id':
        # Convert price to Rupiah for Indonesian
        price_match = re.search(r'\$([\d,.]+)', price_info)
        if price_match:
            price_usd = float(price_match.group(1).replace(',', ''))
            price_idr = f"Rp{price_usd * 16811:,.0f}"  # 1 USD = 16,846 IDR
            
            # Determine price period
            if "per month" in price_info.lower() or "per bulan" in price_info.lower():
                price_period = "per bulan"
            elif "per night" in price_info.lower() or "per malam" in price_info.lower():
                price_period = "per malam"
            elif "2 nights" in price_info.lower() or "2 malam" in price_info.lower():
                price_period = "untuk 2 malam"
            else:
                price_period = "per malam"  # default to per night
                
            price_info = f"Harga mulai dari {price_idr} {price_period}"
        else:
            price_info = f"Informasi harga: {price_info}"
        
        return (f"{title} adalah sebuah {property_type_str} yang berlokasi di {area}, {city}. "
                f"Alamat lengkapnya: {address}. "
                f"Properti ini memiliki {bedroom} kamar tidur, {bathroom} kamar mandi, "
                f"dan dapat menampung hingga {guest_number} tamu. {price_info}")
    else:
        # English version remains the same
        return (f"{title} is a {property_type_str} located in {area}, {city}. "
                f"Full address: {address}. "
                f"This property has {bedroom} bedrooms, {bathroom} bathrooms, "
                f"and can accommodate up to {guest_number} guests. {price_info}")
    
def load_data(file_path=None, df=None):
    if df is None:
        df = pd.read_excel(file_path)
    
    required_columns = [
        'title', 'property_type', 'address', 'city', 
        'area', 'bedroom', 'bathroom', 'guest_number', 
        'price_info'
    ]
    
    # Filter to only include required columns
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Create documents for each property in both languages
    documents = []
    for _, row in df.iterrows():
        metadata = {
            'title': str(row.get('title', '')),
            'property_type': str(row.get('property_type', '')),
            'address': str(row.get('address', '')),
            'city': str(row.get('city', '')),
            'area': str(row.get('area', '')),
            'bedroom': str(row.get('bedroom', '')),
            'bathroom': str(row.get('bathroom', '')),
            'guest_number': str(row.get('guest_number', '')),
            'price_info': str(row.get('price_info', ''))
        }
        
        # Create English version
        doc_en = Document(
            page_content=format_property_info(row, 'en'),
            metadata={**metadata, 'language': 'en'}
        )
        documents.append(doc_en)
        
        # Create Indonesian version
        doc_id = Document(
            page_content=format_property_info(row, 'id'),
            metadata={**metadata, 'language': 'id'}
        )
        documents.append(doc_id)
    
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
    response = chain.invoke({'query': query})
    return response