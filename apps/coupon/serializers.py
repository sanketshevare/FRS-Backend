# serializers.py
from rest_framework import serializers
from .models import CoupounsData

class CoupounDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = CoupounsData
        fields = '__all__'
