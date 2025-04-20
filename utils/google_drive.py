import io
import re
import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import json

def get_drive_service(credentials_info):
    try:
        creds = service_account.Credentials.from_service_account_info(credentials_info)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"Error creating drive service: {e}")
        return None

def get_latest_file(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'",
        fields="files(id, name, createdTime)",
        orderBy="createdTime desc"
    ).execute()

    files = results.get("files", [])
    if not files:
        return None, None

    file_pattern = re.compile(r"data_bukit_vista_(\d{2}-\d{2}-\d{4})\.xlsx")
    latest_file = None
    latest_date = None

    for file in files:
        match = file_pattern.match(file["name"])
        if match:
            file_date = datetime.datetime.strptime(match.group(1), "%d-%m-%Y")
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = file

    return latest_file["id"] if latest_file else None, latest_file["name"] if latest_file else None

def load_latest_data(credentials_info, folder_id):
    service = get_drive_service(credentials_info)
    if not service:
        return None, None

    file_id, file_name = get_latest_file(service, folder_id)
    if not file_id:
        return None, None

    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO(request.execute())
    return pd.read_excel(file_content), file_name