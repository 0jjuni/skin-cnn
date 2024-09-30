from django.urls import path
from .views import GemmaAPIView, CosmeticsFilterAPIView

urlpatterns = [
    path('', GemmaAPIView.as_view(), name='gemma_predict'),
    path('cosmetics', CosmeticsFilterAPIView.as_view(), name='cosmetics_filter'),
]
