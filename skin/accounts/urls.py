# accounts/urls.py
from django.urls import path
from .views import login_api, register_api

urlpatterns = [
    path('login/', login_api, name='login_api'),  # 로그인 API 엔드포인트
    path('register/', register_api, name='register_api'),
]
