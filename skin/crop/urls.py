from django.urls import path
from .views import CropAndSaveAPIView

urlpatterns = [
    path('', CropAndSaveAPIView.as_view(), name='crop_and_save'),
]