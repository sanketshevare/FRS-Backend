"""FRS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
"""
# Django imports
from django.conf.urls import include
from django.contrib import admin
from django.urls import include, re_path, path

from django.contrib.auth import views as auth_views

urlpatterns = [
    # Examples:
    # url(r'^blog/', include('blog.urls', namespace='blog')),

    # provide the most basic login/logout functionality
    re_path(r'^login/$', auth_views.LoginView.as_view(template_name='core/login.html'),
        name='core_login'),
    re_path(r'^logout/$', auth_views.LogoutView.as_view(), name='core_logout'),
    path('api/', include('imageProcess.urls')),  # Include the app's URLs

    # enable the admin interface
    path('admin/', admin.site.urls),
]
