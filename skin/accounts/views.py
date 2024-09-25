# accounts/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
import json

# JWT 토큰 발급 함수
def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

# @csrf_exempt
def login_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')

            # 사용자 인증
            user = authenticate(request, username=username, password=password)

            if user is not None:
                # 인증 성공 시 JWT 토큰 반환
                tokens = get_tokens_for_user(user)
                return JsonResponse({'message': 'Login successful', 'tokens': tokens, 'status': 'success'}, status=200)
            else:
                # 인증 실패
                return JsonResponse({'message': 'Invalid credentials', 'status': 'error'}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'Error: {str(e)}', 'status': 'error'}, status=500)

    return JsonResponse({'message': 'Invalid request method', 'status': 'error'}, status=405)



# accounts/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
import json

#@csrf_exempt
def register_api(request):
    if request.method == 'POST':
        try:
            # 요청으로부터 데이터를 파싱
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            email = data.get('email')

            # 사용자 생성
            if User.objects.filter(username=username).exists():
                return JsonResponse({'message': 'Username already exists', 'status': 'error'}, status=400)

            # 사용자 생성
            user = User.objects.create_user(username=username, password=password, email=email)
            user.save()

            return JsonResponse({'message': 'User registered successfully', 'status': 'success'}, status=201)

        except Exception as e:
            return JsonResponse({'message': f'Error: {str(e)}', 'status': 'error'}, status=500)

    return JsonResponse({'message': 'Invalid request method', 'status': 'error'}, status=405)
