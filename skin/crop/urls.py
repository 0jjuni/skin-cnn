from django.urls import path
from .views import CropAndPredictAPIView
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('', CropAndPredictAPIView.as_view(), name='crop_and_save'),  # csrf_exempt를 as_view()에 적용
]
