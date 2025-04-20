from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import io
import streamlit as st
import json

@st.cache_data
def load_latest_data(credentials):
    try:
        # Create service
        creds = service_account.Credentials.from_service_account_info(credentials)
        service = build("drive", "v3", credentials=creds)
        
        # Get folder ID from secrets
        folder_id = st.secrets["gdrive"]["folder_id"]
        
        # Find latest Excel file
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'",
            fields="files(id, name, createdTime)",
            orderBy="createdTime desc",
            pageSize=1
        ).execute()
        
        files = results.get("files", [])
        if not files:
            st.error("No Excel files found in Google Drive folder")
            return None, None
        
        # Download file
        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO(request.execute())
        
        return pd.read_excel(file_content), files[0]['name']
    
    except Exception as e:
        st.error(f"Error loading data from Google Drive: {e}")
        return None, None