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
from rest_framework_jwt import views as jwt_views

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
    TokenBlacklistView
)

from django.urls import path
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import PasswordResetView
from django.urls import reverse_lazy

# Router for all the APIs
router = routers.SimpleRouter()


class CustomPasswordResetView(PasswordResetView):

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add your custom domain to the context
        context['custom_domain'] = '192.168.1.7'
        print(PasswordResetView)

        return context


urlpatterns = [
    # Examples:
    # url(r'^blog/', include('blog.urls', namespace='blog')),

    # provide the most basic login/logout functionality

    re_path(r'^auth/', include('djoser.urls')),
    re_path(r'^user/login/$', jwt_views.obtain_jwt_token, name='user-login'),
    re_path(r'^login/$', auth_views.LoginView.as_view(template_name='core/login.html'),
            name='core_login'),
    re_path(r'^logout/$', auth_views.LogoutView.as_view(), name='core_logout'),
    # Forgot Password
    path('api/password_reset/', CustomPasswordResetView.as_view(), name='password_reset'),
    path('api/password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('api/reset/<str:uidb64>/<str:token>/', auth_views.PasswordResetConfirmView.as_view(),
         name='password_reset_confirm'),
    path('api/reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),

    path('api/v1/', include(router.urls)),

    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('api/token/blacklist/', TokenBlacklistView.as_view(), name='token_blacklist'),

    path('api/', include('imageProcess.urls')),  # Include the app's URLs
    path('api/', include('coupon.urls')),  # Include the app's URLs

    # enable the admin interface
    path('admin/', admin.site.urls),
]
