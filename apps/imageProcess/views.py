from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.views import APIView
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from sklearn.neighbors import NearestNeighbors
import pickle
import base64
import os
import csv
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
from skimage.segmentation import mark_boundaries
import cv2
from django.shortcuts import render
from FRS.settings.common import MEDIA_ROOT, STATIC_ROOT_FILE
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PIL import Image
import io
from googleapiclient.http import MediaIoBaseDownload

# Load image embeddings and filenames
feature_list = np.array(pickle.load(open(STATIC_ROOT_FILE + '/embeddings.pkl', 'rb')))
filenames = pickle.load(open(STATIC_ROOT_FILE + '/filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


def get_recommendations(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    distances, indices = neighbors.kneighbors([normalized_result])
    return indices[0][1:5]


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


class Process_image(APIView):
    base = LimeImageExplainer()
    text_explainer = LimeTextExplainer()

    def post(self, request):
        img = request.FILES.get("image")
        file_content = ContentFile(img.read())
        file_name = default_storage.save('uploaded_image.jpg', file_content)
        uploaded_image_path = os.path.join(MEDIA_ROOT, file_name)

        # Get recommendations based on the uploaded image
        recommendations = get_recommendations(uploaded_image_path)

        # Fetch images from Google Drive
        fetched_images = fetch_images_from_google_drive()

        # Get recommendations based on fetched images
        recommended_images = [fetched_images[idx] for idx in recommendations]

        base64_images = []
        os.remove(uploaded_image_path)
        # Additional code for LIME explanations and metadata
        explanations = []
        metadata_for_recommended = []

        csv_file_path = "static/dataset/nike-dataset.csv"

        # Load CSV data
        with open(csv_file_path, 'r', encoding='latin1') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            csv_data = [row for row in csv_reader]

        csv_lookup = {entry['image_name']: entry for entry in csv_data}

        for r_image in recommended_images:
            # Get LIME explanation for each recommended image

            # Get metadata for each recommended image
            explanation = self.get_lime_explanation(r_image)
            explanations.append(explanation)
            metadata = csv_lookup.get(os.path.basename(r_image), {})
            metadata_for_recommended.append(metadata)

            base64_image = image_to_base64(r_image)
            base64_images.append(base64_image)

        return JsonResponse(
            {'recommendations': base64_images, 'explanations': explanations, 'metadata': metadata_for_recommended})

    # Other methods remain unchanged...


    def get_lime_explanation(self, image_path):
        img_path = os.path.join(MEDIA_ROOT, image_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        # Define a function to predict using your model
        predict_fn = lambda x: model.predict(x)

        # Use self.base instead of base
        ret_exp = self.base.explain_instance(preprocessed_img[0], predict_fn, top_labels=1, hide_color=0,
                                             num_samples=1)

        local_pred_list = ret_exp.local_pred.tolist()

        # Get the explanation image and mask
        temp, mask = ret_exp.get_image_and_mask(ret_exp.top_labels[0], positive_only=True, num_features=1500,
                                                hide_rest=True)

        # Convert the image data to a NumPy array and adjust the range
        img_boundaries = (mark_boundaries(temp / 2 + 0.5, mask) * 255).astype(np.uint8)

        # Construct the full file path
        explanation_image_path = os.path.join(MEDIA_ROOT, "explanation_image.png")
        cv2.imwrite(explanation_image_path, img_boundaries)
        explanation_encoded_image = image_to_base64(explanation_image_path)
        # Save the image with boundaries using OpenCV
        # text_explanation = self.get_text_explanation(image_path)
        pred_proba = model.predict(preprocessed_img)[0][ret_exp.top_labels[0]]

        response_data = {
            "label": int(ret_exp.top_labels[0]),  # Convert to int
            "local_pred_shape": local_pred_list,
            "confidence": float(pred_proba),
            # Add other relevant information as needed
        }
        # Return the dictionary, not JsonResponse directly
        return response_data

    def get_text_explanation(self, image_path):
        img_path = os.path.join(MEDIA_ROOT, image_path)
        # Define a function to predict using your model
        predict_fn = lambda x: model.predict(x)

        text_explanation = self.text_explainer.explain_instance(img_path, predict_fn, num_samples=10)
        explanation_text = text_explanation.as_list()

        # Convert the explanation into a human-readable format
        text_result = []
        for feature, weight in explanation_text:
            text_result.append(f"{feature}: {weight}")

        return "\n".join(text_result)


def fetch_images_from_google_drive():
    # Path to the service account JSON key file
    SERVICE_ACCOUNT_FILE = STATIC_ROOT_FILE + '/client_secrets.json'
    credentials = service_account.Credentials.from_service_account_file(STATIC_ROOT_FILE + '/client_secrets.json')

    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive']
    )

    # Build the Drive service
    drive_service = build('drive', 'v3', credentials=credentials)

    # ID of the Google Drive folder containing the images
    folder_id = '1rZjbihmfroRmcn1PgxV_qVhHL8bszlk5'

    # Fetch images from the specified folder
    results = drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                                         fields="files(id, name)").execute()
    images = []

    for file in results.get('files', []):
        file_id = file.get('id')
        file_name = file.get('name')

        # Download the image file content
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        # Save the downloaded image
        image_path = os.path.join(MEDIA_ROOT, file_name)
        with open(image_path, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())

        images.append(image_path)

    return images


def resetPassword(request, uid, token):
    print(uid)
    print(token)
    context = {
        'uid': uid,
        'token': token
    }
    return render(request, 'registration/password_reset_email.html', context)


def resetPassword1(request):
    print("resetting password")
    if request.method == 'POST':
        print(request.POST)
        uid = request.POST.get('uid')
        token = request.POST.get('token')
        print(uid)
        print(token)
    return render(request, 'registration/password_reset_email.html')
