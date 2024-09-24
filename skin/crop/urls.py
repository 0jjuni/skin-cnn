from django.urls import path
from .views import CropAndPredictAPIView

urlpatterns = [
    path('', CropAndPredictAPIView.as_view(), name='crop_and_save'),
]