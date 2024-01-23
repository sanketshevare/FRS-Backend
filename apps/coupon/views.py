# views.py
from rest_framework import generics
from .models import CoupounsData
from .serializers import CoupounDataSerializer

class CoupounDataListCreateView(generics.ListCreateAPIView):
    queryset = CoupounsData.objects.all()
    serializer_class = CoupounDataSerializer
