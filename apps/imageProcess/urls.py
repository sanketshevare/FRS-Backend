

# imageProcess/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('recommend', views.Process_image.as_view(), name='recommend'),
    path('password/reset/confirm/<str:uid>/<str:token>', views.resetPassword, name='test'),

    path('resetpassword', views.resetPassword1, name='resetpassword'),
    # path('auth/register/', views.CustomUserCreateView.as_view(), name='user-register'),

]
