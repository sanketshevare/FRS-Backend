"""FRS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
"""
# Django imports
from django.conf.urls import include
from django.contrib import admin
from django.urls import include, re_path, path
from rest_framework import routers
from django.contrib.auth import views as auth_views

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
    TokenBlacklistView
)
# Router for all the APIs
router = routers.SimpleRouter()

urlpatterns = [
    # Examples:
    # url(r'^blog/', include('blog.urls', namespace='blog')),

    # provide the most basic login/logout functionality
    re_path(r'^login/$', auth_views.LoginView.as_view(template_name='core/login.html'),
        name='core_login'),
    re_path(r'^logout/$', auth_views.LogoutView.as_view(), name='core_logout'),
    
    path('api/v1/', include(router.urls)),

    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('api/token/blacklist/', TokenBlacklistView.as_view(), name='token_blacklist'),



    path('api/', include('imageProcess.urls')),  # Include the app's URLs

    # enable the admin interface
    path('admin/', admin.site.urls),
]
