import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from os.path import dirname, abspath
from tqdm import tqdm
import pickle
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PIL import Image
import io
from google.oauth2 import service_account

# Path to the service account JSON key file
SERVICE_ACCOUNT_FILE = '../static/client_secrets.json'
credentials = service_account.Credentials.from_service_account_file('../static/client_secrets.json')

# Authenticate using service account credentials
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive']
)

# Build the Drive service
drive_service = build('drive', 'v3', credentials=credentials)

# ID of the Google Drive folder containing the images
folder_id = '1rZjbihmfroRmcn1PgxV_qVhHL8bszlk5'

# List all files in the specified folder
results = drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                                     fields="files(id, name, mimeType)").execute()
file_list = results.get('files', [])

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Extract features for each image in the Google Drive folder
feature_list = []
filenames = []

for file in tqdm(file_list):
    if file['mimeType'] == 'image/jpeg' or file['mimeType'] == 'image/png':
        # Fetch image content from Google Drive
        img_content = drive_service.files().get_media(fileId=file['id']).execute()
        img_bytes = io.BytesIO(img_content)
        img = image.img_to_array(Image.open(img_bytes).convert('RGB').resize((224, 224)))
        expanded_img_array = np.expand_dims(img, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Append the features and filename to lists
        feature_list.append(normalized_result)
        filenames.append(file['name'])


DJANGO_ROOT = dirname(dirname(abspath(__file__)))

# Save features and filenames to pickle files
pickle.dump(feature_list, open(DJANGO_ROOT + '/static/embeddings.pkl', 'wb'))
pickle.dump(filenames, open(DJANGO_ROOT + '/static/filenames.pkl', 'wb'))

print("Features extracted and saved successfully.")
