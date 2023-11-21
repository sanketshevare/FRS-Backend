

# imageProcess/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('recommend', views.Process_image.as_view(), name='recommend'),
]
