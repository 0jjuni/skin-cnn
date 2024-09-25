from rest_framework import serializers
from accounts.models import PredictionResult

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = [
            'forehead_age_prediction',
            'forehead_age_probability',
            'left_cheek_age_prediction',
            'left_cheek_age_probability',
            'right_cheek_age_prediction',
            'right_cheek_age_probability',
            'forehead_pigmentation_prediction',
            'forehead_pigmentation_probability',
            'forehead_moisture_prediction',
            'forehead_moisture_probability',
            'left_cheek_moisture_prediction',
            'left_cheek_moisture_probability',
            'right_cheek_moisture_prediction',
            'right_cheek_moisture_probability',
            'created_at'
        ]
