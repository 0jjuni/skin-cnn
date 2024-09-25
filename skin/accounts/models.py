from django.contrib.auth.models import User
from django.db import models

# DB 모델 정의
class PredictionResult(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True)  # 사용자와 연결, 1:1 관계로 설정

    # 나이 예측 결과
    forehead_age_prediction = models.IntegerField()
    forehead_age_probability = models.FloatField()
    left_cheek_age_prediction = models.IntegerField()
    left_cheek_age_probability = models.FloatField()
    right_cheek_age_prediction = models.IntegerField()
    right_cheek_age_probability = models.FloatField()

    # 색소 침착 예측 결과 (이마)
    forehead_pigmentation_prediction = models.IntegerField()
    forehead_pigmentation_probability = models.FloatField()

    # 수분 예측 결과
    forehead_moisture_prediction = models.IntegerField()
    forehead_moisture_probability = models.FloatField()
    left_cheek_moisture_prediction = models.IntegerField()
    left_cheek_moisture_probability = models.FloatField()
    right_cheek_moisture_prediction = models.IntegerField()
    right_cheek_moisture_probability = models.FloatField()

    # 스킨 타입 예측 결과
    forehead_skin_prediction = models.IntegerField(null=True, blank=True)
    forehead_skin_probability = models.FloatField(null=True, blank=True)
    left_cheek_skin_prediction = models.IntegerField(null=True, blank=True)
    left_cheek_skin_probability = models.FloatField(null=True, blank=True)
    right_cheek_skin_prediction = models.IntegerField(null=True, blank=True)
    right_cheek_skin_probability = models.FloatField(null=True, blank=True)

    # 생성 시각
    created_at = models.DateTimeField(auto_now_add=True)
