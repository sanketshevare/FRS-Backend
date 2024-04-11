# urls.py
from django.urls import path
from .views import CoupounDataListCreateView

urlpatterns = [
    path('coupoundata/', CoupounDataListCreateView.as_view(), name='coupoundata-list-create'),
    # Add other URL patterns as needed
]
