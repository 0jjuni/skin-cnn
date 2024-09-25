from django.urls import path
from .views import GemmaAPIView

urlpatterns = [
    path('', GemmaAPIView.as_view(), name='gemma_predict'),
]
