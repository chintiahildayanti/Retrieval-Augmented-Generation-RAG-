import streamlit as st
from utils.google_drive import load_latest_data
from utils.lang_funcs import load_data, split_docs, load_embedding_model, create_embeddings, load_qa_chain
from langchain.llms import Ollama
from langchain import PromptTemplate
import base64
import json

# Konfigurasi Aplikasi
st.set_page_config(page_title="Property Chatbot", layout="wide")

# Fungsi untuk mendapatkan credentials
def get_credentials():
    if "GDRIVE_CREDENTIALS_BASE64" in st.secrets:
        creds_json = base64.b64decode(st.secrets["GDRIVE_CREDENTIALS_BASE64"]).decode('utf-8')
        return json.loads(creds_json)
    elif "credentials" in st.secrets["gdrive"]:
        return json.loads(st.secrets["gdrive"]["credentials"])
    else:
        st.error("Google Drive credentials not found")
        st.stop()

# Inisialisasi sistem
@st.cache_resource
def initialize_system():
    try:
        # 1. Load data
        creds = get_credentials()
        df, _ = load_latest_data(creds)
        if df is None:
            st.error("Failed to load data from Google Drive")
            st.stop()
        
        # 2. Process data
        docs = load_data(df=df)
        documents = split_docs(documents=docs)
        
        # 3. Setup embeddings
        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
        vectorstore = create_embeddings(documents, embed)
        
        # 4. Setup LLM
        llm = Ollama(model="orca-mini", temperature=0)
        
        # 5. Create prompt template
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
        
        # 6. Create QA chain
        return load_qa_chain(vectorstore.as_retriever(), llm, prompt)
    
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

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
            try:
                # Dapatkan response
                response = chain({'query': query})
                
                # Tampilkan hasil
                st.subheader("Hasil Pencarian")
                st.write(response['result'])
                
                # Tampilkan properti terkait
                if 'source_documents' in response:
                    st.subheader("Properti Terkait")
                    for i, doc in enumerate(response['source_documents'][:3]):
                        meta = doc.metadata
                        with st.expander(f"{i+1}. {meta.get('title', 'Properti')} - {meta.get('city', '')}"):
                            cols = st.columns([1,2])
                            cols[0].metric("Kamar Tidur", meta.get('bedroom', '-'))
                            cols[0].metric("Kamar Mandi", meta.get('bathroom', '-'))
                            cols[1].write(f"**Alamat**: {meta.get('address', '')}")
                            cols[1].write(f"**Tipe**: {meta.get('property_type', '')}")
                            cols[1].write(f"**Luas**: {meta.get('area', '')} m²")
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()