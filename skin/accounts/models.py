from django.contrib.auth.models import User
from django.db import models

# DB 모델 정의
class PredictionResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 사용자와 연결
    forehead_age_prediction = models.IntegerField()
    forehead_age_probability = models.FloatField()
    left_cheek_age_prediction = models.IntegerField()
    left_cheek_age_probability = models.FloatField()
    right_cheek_age_prediction = models.IntegerField()
    right_cheek_age_probability = models.FloatField()
    forehead_pigmentation_prediction = models.IntegerField()
    forehead_pigmentation_probability = models.FloatField()
    forehead_moisture_prediction = models.IntegerField()
    forehead_moisture_probability = models.FloatField()
    left_cheek_moisture_prediction = models.IntegerField()
    left_cheek_moisture_probability = models.FloatField()
    right_cheek_moisture_prediction = models.IntegerField()
    right_cheek_moisture_probability = models.FloatField()

    # 추가적인 예측 결과 필드 추가 가능
    created_at = models.DateTimeField(auto_now_add=True)
