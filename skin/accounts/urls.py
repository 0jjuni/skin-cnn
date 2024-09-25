# accounts/urls.py
from django.urls import path
from .views import register_api
# urls.py
from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    # JWT 관련 엔드포인트
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('register/', register_api, name='register_api'),
]
