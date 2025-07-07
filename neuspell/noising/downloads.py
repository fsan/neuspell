# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

import os
import re

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Check if we got a virus scan warning page
    if response.headers.get('content-type', '').startswith('text/html'):
        html_content = response.text
        
        if 'Google Drive can\'t scan this file for viruses' in html_content:
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
            if uuid_match:
                uuid = uuid_match.group(1)
                download_url = f"https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t&uuid={uuid}"
                
                response = session.get(download_url, stream=True)
                
                if response.headers.get('content-type', '').startswith('text/html'):
                    authuser_match = re.search(r'name="authuser" value="([^"]*)"', html_content)
                    authuser = authuser_match.group(1) if authuser_match else "0"
                    
                    download_url = f"https://drive.usercontent.google.com/download?id={id}&export=download&authuser={authuser}&confirm=t&uuid={uuid}"
                    response = session.get(download_url, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
        return True
    else:
        print(f"{path_} already exists")
    return False

