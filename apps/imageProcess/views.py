
from django.http import JsonResponse
from django.http import HttpResponse
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
# Create your views here.
import requests
from rest_framework.views import APIView
from rest_framework import permissions

import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pickle
import base64

from FRS.settings.common import MEDIA_ROOT, STATIC_ROOT, STATICFILES_DIRS, STATIC_ROOT_FILE

# Load image embeddings and filenames
feature_list = np.array(pickle.load(open(STATIC_ROOT_FILE +'/embeddings.pkl', 'rb')))
filenames = pickle.load(open(STATIC_ROOT_FILE +'/filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


def get_recommendations(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    distances, indices = neighbors.kneighbors([normalized_result])
    return indices[0][1:6]


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


def save_base64_image(base64_string, output_path):
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as image_file:
        image_file.write(image_data)

class Process_image(APIView):
    # permission_classes = [permissions.IsAuthenticated]

    def post(self, request):

        img = request.FILES.get("image")
        file_content = ContentFile(img.read())
        file_name = default_storage.save('uploaded_image.jpg', file_content)

        # Create
        recommendations = get_recommendations(MEDIA_ROOT+'/uploaded_image.jpg')
        recommended_images = [filenames[idx] for idx in recommendations]
        base64_images = []
        for r_image in recommended_images:
            base64_image = image_to_base64(r_image)
            base64_images.append(base64_image)

        return JsonResponse({'recommendations': base64_images})


class Test(APIView):
    # permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        print("Hello from test")

        return JsonResponse({'test': "test done..."})


# views.py

from django.core.cache import cache
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class LogoutView(APIView):
    def post(self, request):
        try:
            refresh_token = request.data["token"]
            token = RefreshToken(refresh_token)
            token.blacklist()
            # Optionally store the invalidated token in cache
            cache.set(refresh_token, "", timeout=60 * 60 * 24 * 30)  # Set a timeout if needed
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
