import streamlit as st
from lang_funcs import (
    load_data, split_docs, load_embedding_model, 
    create_embeddings, load_qa_chain, detect_query_language
)
from langchain_community.llms import Ollama
from langchain import PromptTemplate
import base64
import json
import re
import pandas as pd
import time

# Konfigurasi Aplikasi
st.set_page_config(page_title="Chatbot Properti", layout="wide")

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


from langchain.schema import Document
def load_data(df):
    documents = []
    for idx, row in df.iterrows():
        bedroom = 0 if pd.isna(row.get('bedroom')) else int(float(row.get('bedroom', 0)))
        guest_number = 0 if pd.isna(row.get('guest_number')) else int(float(row.get('guest_number', 0)))

        metadata = {
            "title": row.get("title", ""),
            "property_type": row.get("property_type", ""),
            "address": row.get("address", ""),
            "city": row.get("city", ""),
            "area": row.get("area", ""),
            "bedroom": bedroom,
            "bathroom": 0 if pd.isna(row.get('bathroom')) else row.get('bathroom', 0),
            "guest_number": guest_number,
            "price_info": row.get("price_info", ""),
            "property_status": row.get("property_status", ""),
            "address_detail": row.get("address_detail", ""),
            "tags": row.get("tags", ""),
            "price": row.get("price", 0),
            "image_url": row.get("image_url", ""),
            # tambah apapun yang kamu rasa penting
        }
        content = f"{row.get('title', '')} - {row.get('address', '')} - {row.get('area', '')}"
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def format_property_response(property_data):
    title = property_data.get('title', 'Properti')
    property_type = property_data.get('cleaned_property_type', property_data.get('property_type', ''))
    property_type_str = f"'{property_type}'" if property_type else ''
    address = property_data.get('address', '')
    city = property_data.get('city', '')
    area = property_data.get('cleaned_area', property_data.get('area', ''))
    # Handle potential None or NaN values
    bedroom = property_data.get('bedroom', 0)
    bathroom = property_data.get('bathroom', 0)
    guest_number = property_data.get('guest_number', 0)
    
    # Convert to int if not already
    try:
        bedroom = int(bedroom) if bedroom is not None and not pd.isna(bedroom) else 0
        bathroom = int(bathroom) if bathroom is not None and not pd.isna(bathroom) else 0
        guest_number = int(guest_number) if guest_number is not None and not pd.isna(guest_number) else 0
    except (ValueError, TypeError):
        bedroom = bathroom = guest_number = 0
        
    price_info = property_data.get('price_info', '')
    
    # Format harga
    price_text = "Harga mulai dari "
    if pd.notna(price_info) and str(price_info).strip():
        price_match = re.search(r'\$([\d,.]+)', str(price_info))
        if price_match:
            price_usd = float(price_match.group(1).replace(',', ''))
            price_idr = f"Rp{price_usd * 16811:,.0f}"
            
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

    return (f"{title} adalah sebuah {property_type_str} yang berlokasi di {area}, {city}. "
            f"Alamat lengkapnya terletak di {address}. "
            f"Properti ini memiliki {bedroom} kamar tidur, {bathroom} kamar mandi, "
            f"dan dapat menampung hingga {guest_number} tamu. {price_text}")

@st.cache_resource
@st.cache_resource
def initialize_system():
    try:
        # Load data langsung dari file lokal
        file_path = 'data_bukit_vista.xlxs'
        df = pd.read_excel(file_path)
        
        if df is None:
            st.error("Gagal memuat data dari file lokal")
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
        llm = Ollama(model="orca-mini:7b", temperature=0.1)
        
        return vectorstore, llm
    
    except Exception as e:
        st.error(f"Error inisialisasi: {str(e)}")
        st.stop()

def main():
    st.title("🤖 TANYA - AI Assistant")
    st.caption("Tanyakan tentang properti liburan Anda di Bali dan Yogyakarta")
    
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
                    Anda adalah asisten properti Bali yang ramah, profesional, dan sangat membantu.
                    Gunakan HANYA informasi yang tersedia dari konteks di bawah ini untuk memberikan jawaban.
                    Jangan mengarang atau mengasumsikan informasi di luar konteks.
                    Jika tidak menemukan properti yang cocok, katakan dengan sopan dan tawarkan bantuan lebih lanjut.

                    Ketika menjawab:
                    1. Mulai dengan sapaan hangat.
                    2. Jelaskan properti yang relevan dengan detail berikut:
                    - Nama properti (title)
                    - Jenis properti (cleaned_property_type atau property_type)
                    - Lokasi: area dan city
                    - Jumlah kamar tidur dan kamar mandi
                    - Kapasitas jumlah tamu (guest_number)
                    - Harga sewa dan periode sewa (price_info, rental_period)
                    - Informasi tambahan yang relevan (tags, garage, property_status) jika tersedia.
                    3. Jika informasi tidak lengkap, berikan sebaik mungkin dari data yang ada, tanpa mengarang.
                    4. Tawarkan untuk memberikan rekomendasi lain atau membantu pencarian lebih lanjut.

                    Jika pertanyaan pengguna tidak menyebutkan:
                    - Lokasi (city atau area): tanyakan dengan sopan lokasi yang diinginkan.
                    - Jumlah tamu, kamar tidur, atau anggaran: tawarkan pilihan sesuai konteks, atau tanyakan untuk memperjelas kebutuhan.

                    Gunakan gaya bahasa yang ramah, sopan, dan antusias.

                    Contoh Respons:
                    "Halo! Saya menemukan beberapa villa yang mungkin cocok untuk Anda! ✨
                    - **Villa Sundara**: Villa mewah di area Ubud, 3 kamar tidur, 2 kamar mandi, cocok untuk 6 tamu. Harga Rp3.000.000/malam.
                    - **Villa Harmony**: Terletak di Seminyak, 2 kamar tidur, kapasitas 4 tamu. Harga Rp2.500.000/malam.

                    Apakah Anda ingin saya bantu dengan lebih banyak pilihan atau detail lebih lanjut?"

                    ---

                    Konteks properti:
                    {context}

                    Pertanyaan pengguna:
                    {question}

                    Respons:
                    """
                    
                    template = PromptTemplate.from_template(prompt_template)
                    chain = load_qa_chain(vectorstore.as_retriever(), llm, template)
                    response = chain({'query': prompt})
                    
                    # Format the response
                    if 'source_documents' in response:
                        properties = [format_property_response(doc.metadata) for doc in response['source_documents'][:3]]
                        message_parts = [
                            f"Halo! Berdasarkan permintaan Anda tentang '{prompt}', saya menemukan beberapa pilihan properti ini untuk Anda :"
                        ]
                        
                        for prop in properties:
                            message_parts.append(f"\n\n{prop}")
                        
                        message_parts.append("\n\nApakah Anda ingin informasi tentang properti lain?")
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