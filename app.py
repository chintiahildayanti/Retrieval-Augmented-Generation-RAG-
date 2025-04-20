import streamlit as st
from utils.google_drive import load_latest_data
from utils.lang_funcs import load_data, split_docs, load_embedding_model, create_embeddings, load_qa_chain, get_response
from langchain.llms import Ollama
from langchain import PromptTemplate
import pandas as pd

# Konfigurasi Aplikasi
st.set_page_config(page_title="Property Chatbot", layout="wide")

# Fungsi untuk inisialisasi sistem
@st.cache_resource
def initialize_system():
    # 1. Load data dari Google Drive
    df, _ = load_latest_data()
    if df is None:
        st.error("Gagal memuat data dari Google Drive")
        st.stop()
    
    # 2. Proses data
    docs = load_data(df=df)
    documents = split_docs(documents=docs)
    
    # 3. Setup embedding dan vectorstore
    embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
    vectorstore = create_embeddings(documents, embed)
    
    # 4. Setup LLM
    llm = Ollama(model="orca-mini", temperature=0)
    
    # 5. Buat prompt template
    template = """
    ### System:
    Anda adalah asisten properti yang membantu menemukan properti terbaik untuk klien.
    Gunakan hanya informasi dari context. Jika tidak tahu, katakan tidak tahu.

    ### Context:
    {context}

    ### Question:
    {question}

    ### Response:
    """
    prompt = PromptTemplate.from_template(template)
    
    # 6. Buat QA chain
    return load_qa_chain(vectorstore.as_retriever(), llm, prompt)

# Main App
def main():
    st.title("🏠 Property Chatbot")
    st.write("Tanyakan tentang properti yang tersedia")
    
    # Inisialisasi sistem
    chain = initialize_system()
    
    # Input pengguna
    query = st.text_input("Apa yang ingin Anda ketahui tentang properti?", "")
    
    if query:
        with st.spinner("Mencari informasi..."):
            # Dapatkan response
            response = chain({'query': query})
            
            # Tampilkan hasil
            st.subheader("Hasil Pencarian")
            st.write(response['result'])
            
            # Tampilkan properti terkait (dari metadata)
            if 'source_documents' in response:
                st.subheader("Properti Terkait")
                for doc in response['source_documents'][:3]:  # Batasi 3 hasil
                    meta = doc.metadata
                    with st.expander(f"{meta.get('title', 'Properti')} - {meta.get('city', '')}"):
                        cols = st.columns([1,2])
                        cols[0].metric("Kamar Tidur", meta.get('bedroom', '-'))
                        cols[0].metric("Kamar Mandi", meta.get('bathroom', '-'))
                        cols[1].write(f"**Alamat**: {meta.get('address', '')}")
                        cols[1].write(f"**Tipe**: {meta.get('property_type', '')}")
                        cols[1].write(f"**Luas**: {meta.get('area', '')} m²")

if __name__ == "__main__":
    main()