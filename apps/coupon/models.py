# models.py
from django.db import models
from datetime import datetime


class CoupounsData(models.Model):
    code = models.CharField(unique=True, max_length=255, blank=False, null=False)
    brand = models.CharField(unique=False, max_length=255, blank=False, null=False,
                             default='')
    discount = models.CharField(unique=False, max_length=255, blank=False, null=False,  default='0')
    min_amount = models.CharField(unique=False, max_length=255, blank=False, null=False,  default='0')
    exp_date = models.CharField(unique=False, max_length=255, blank=True, null=True,  default=str(datetime.now()))
    desc = models.CharField(unique=False, max_length=255, blank=False, null=False,  default='0')

    def __str__(self):
        return self.name
