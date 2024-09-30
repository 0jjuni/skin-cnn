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
            'forehead_pores_prediction',
            'forehead_pores_probability',
            'left_cheek_pores_prediction',
            'left_cheek_pores_probability',
            'right_cheek_pores_prediction',
            'right_cheek_pores_probability',

            'created_at'
        ]


from rest_framework import serializers
from .models import Cosmetics  # Cosmetics 모델 import

class CosmeticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cosmetics
        fields = [
            'product_id',
            'product_name',
            'brand_name',
            'price',
            'description',
            'image_path',
            'category_id',
            'created_at',
            'updated_at',
            'cosmetics_type',
            'age_type',
            'skin_type',
            'moisture_type',
            'pigmentation_type',
            'pores_type',
        ]
