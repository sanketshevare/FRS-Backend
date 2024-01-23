from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext as _
import uuid


class CustomUser(AbstractUser):
    # Demographic Fields required for Bahmni
    middle_name = models.CharField(max_length=200, null=True, blank=True)
    gender = models.CharField(max_length=200, null=True, blank=True, verbose_name=_("Gender"))
    mobile_number = models.CharField(max_length=10, unique=True, null=True)
    birth_date = models.DateField(null=True, blank=True, verbose_name=_("Date of birth"))

    # Location specific fields
    address1 = models.CharField(max_length=200, null=True, blank=True)
    address2 = models.CharField(max_length=200, null=True, blank=True)

    # Bahmni Specific Identifiers

    # JWT Secret
    jwt_secret = models.CharField(max_length=200, default=uuid.uuid4)
    uuid = models.UUIDField(default=uuid.uuid4, editable=False)
    session_id = models.CharField(max_length=500, null=True, blank=True)

    # Datetime model fields
    datetime_created = models.DateTimeField(auto_now_add=True)
    datetime_updated = models.DateTimeField(auto_now=True)
    email = models.EmailField(unique=True)  # Ensure email is unique

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'username', 'password', 'mobile_number', 'birth_date', 'gender']


def jwt_get_secret_key(user_model):
    return user_model.jwt_secret
