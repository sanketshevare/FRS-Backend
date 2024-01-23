
from djoser.serializers import UserSerializer
from django.contrib.auth.models import AbstractUser

class CustomUserSerializer(UserSerializer):
    class Meta:
        model = AbstractUser
        fields = ('id', 'username', 'email', 'first_name', 'last_name')  # Add 'first_name' and other fields as needed
