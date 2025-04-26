import streamlit as st
from utils.google_drive import load_latest_data
from utils.lang_funcs import (
    load_data, split_docs, load_embedding_model, 
    create_embeddings, load_qa_chain, detect_query_language
)
from langchain.llms import Ollama
from langchain import PromptTemplate
import base64
import json
import re
import pandas as pd
import time

# Konfigurasi Aplikasi
st.set_page_config(page_title="Chatbot Properti", layout="wide")

def get_credentials():
    if "GDRIVE_CREDENTIALS_BASE64" in st.secrets:
        creds_json = base64.b64decode(st.secrets["GDRIVE_CREDENTIALS_BASE64"]).decode('utf-8')
        return json.loads(creds_json)
    elif "credentials" in st.secrets["gdrive"]:
        return json.loads(st.secrets["gdrive"]["credentials"])
    else:
        st.error("Kredensial Google Drive tidak ditemukan")
        st.stop()

def format_property_response(property_data):
    title = property_data.get('title', 'Properti')
    property_type = property_data.get('cleaned_property_type', property_data.get('property_type', ''))
    address = property_data.get('address', '')
    city = property_data.get('city', '')
    area = property_data.get('cleaned_area', property_data.get('area', ''))
    bedroom = property_data.get('bedroom', '')
    bathroom = property_data.get('bathroom', '')
    guest_number = property_data.get('guest_number', '')
    price_info = property_data.get('price_info', '')
    
    # Format harga
    price_text = "Harga: "
    if pd.notna(price_info) and str(price_info).strip():
        price_match = re.search(r'\$([\d,.]+)', str(price_info))
        if price_match:
            price_usd = float(price_match.group(1).replace(',', ''))
            price_idr = f"Rp{price_usd * 15000:,.0f}"
            
            if "per month" in str(price_info).lower():
                price_text += f"{price_idr}/bulan"
            elif "per night" in str(price_info).lower():
                price_text += f"{price_idr}/malam"
            elif "2 nights" in str(price_info).lower():
                price_text += f"{price_idr}/2 malam"
            else:
                price_text += price_idr
        else:
            price_text += str(price_info)
    else:
        price_text = ""

    return {
        'title': title,
        'description': (
            f"✨ **{title}**\n\n"
            f"🏡 **Tipe:** {property_type}\n"
            f"📍 **Lokasi:** {area}, {city}\n"
            f"🏠 **Kamar:** {bedroom} tidur, {bathroom} mandi (Maks {guest_number} tamu)\n"
            f"💰 **{price_text}**\n"
            f"🏷️ **Alamat:** {address}"
        ),
        'short': f"{title} ({area}, {city}) - {price_text}"
    }

@st.cache_resource
def initialize_system():
    try:
        creds = get_credentials()
        df, _ = load_latest_data(creds)
        if df is None:
            st.error("Gagal memuat data dari Google Drive")
            st.stop()
        
        required_columns = [
            'title', 'property_type', 'cleaned_property_type',
            'address', 'city', 'area', 'cleaned_area',
            'bedroom', 'bathroom', 'guest_number', 'price_info'
        ]
        
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]
        
        docs = load_data(df=df)
        documents = split_docs(documents=docs)
        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
        vectorstore = create_embeddings(documents, embed)
        llm = Ollama(model="orca-mini", temperature=0.7)  # Higher temp for more creative responses
        
        return vectorstore, llm
    
    except Exception as e:
        st.error(f"Error inisialisasi: {str(e)}")
        st.stop()

def display_thinking_steps():
    steps = [
        "Menganalisis permintaan Anda",
        "Mencari properti yang cocok",
        "Memverifikasi informasi",
        "Menyiapkan rekomendasi"
    ]
    
    placeholder = st.empty()
    for step in steps:
        with placeholder.container():
            st.markdown(f"✓ {step}")
            time.sleep(0.5)
    placeholder.empty()

def main():
    st.title("🏠 Asisten Properti Bali")
    st.caption("Tanyakan tentang properti liburan Anda di Bali")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    vectorstore, llm = initialize_system()
    
    if prompt := st.chat_input("Apa yang Anda cari dalam properti liburan?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Memproses..."):
                try:
                    # Show thinking steps
                    display_thinking_steps()
                    
                    language = detect_query_language(prompt)
                    
                    prompt_template = """
                    Anda adalah asisten properti Bali yang ramah dan membantu. 
                    Gunakan HANYA informasi dari konteks di bawah ini.
                    Jika tidak tahu, jangan membuat informasi.
                    
                    Format respons:
                    1. Awali dengan sapaan hangat
                    2. Jelaskan properti yang cocok dengan detail
                    3. Sertakan informasi harga yang jelas
                    4. Akhiri dengan tawaran bantuan lebih lanjut
                    
                    Contoh respons:
                    "Halo! Saya menemukan villa bagus untuk Anda...
                    [Detail properti]
                    Apakah Anda ingin informasi lebih lanjut?"
                    
                    Konteks:
                    {context}
                    
                    Pertanyaan:
                    {question}
                    
                    Respons:
                    """
                    
                    template = PromptTemplate.from_template(prompt_template)
                    chain = load_qa_chain(vectorstore.as_retriever(), llm, template)
                    response = chain({'query': prompt})
                    
                    # Format the response
                    if 'source_documents' in response:
                        properties = [format_property_response(doc.metadata) for doc in response['source_documents'][:3]]
                        
                        # Build response message
                        message_parts = [
                            f"Halo! Berdasarkan permintaan Anda tentang '{prompt}', saya menemukan beberapa pilihan properti:"
                        ]
                        
                        for prop in properties:
                            message_parts.append(f"\n\n🌟 **{prop['title']}**")
                            message_parts.append(prop['description'])
                        
                        message_parts.append("\n\nApakah Anda ingin informasi lebih detail tentang salah satu properti di atas?")
                        
                        full_response = "\n".join(message_parts)
                    else:
                        full_response = "Maaf, saya tidak menemukan properti yang cocok dengan kriteria Anda. Bisakah Anda menjelaskan lebih detail?"
                    
                    # Simulate typing effect
                    response_placeholder = st.empty()
                    full_text = ""
                    for chunk in full_response.split(" "):
                        full_text += chunk + " "
                        response_placeholder.markdown(full_text + "▌")
                        time.sleep(0.05)
                    response_placeholder.markdown(full_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_text})
                    
                except Exception as e:
                    error_msg = f"Maaf, terjadi error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()