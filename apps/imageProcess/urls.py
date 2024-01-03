

# imageProcess/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('recommend', views.Process_image.as_view(), name='recommend'),
    path('test', views.Test.as_view(), name='test'),
    path('logout', views.LogoutView.as_view(), name='token_logout'),



]
