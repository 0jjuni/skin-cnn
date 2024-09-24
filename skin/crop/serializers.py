from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    # 나이 예측 결과 (이마, 왼쪽 볼, 오른쪽 볼)
    forehead_age_prediction = serializers.IntegerField()
    forehead_age_probabilities = serializers.ListField(child=serializers.FloatField())

    left_cheek_age_prediction = serializers.IntegerField()  # 필드명 수정
    left_cheek_age_probabilities = serializers.ListField(child=serializers.FloatField())

    right_cheek_age_prediction = serializers.IntegerField()  # 필드명 수정
    right_cheek_age_probabilities = serializers.ListField(child=serializers.FloatField())

    # 색소 침착 예측 결과
    forehead_pigmentation_prediction = serializers.IntegerField()
    forehead_pigmentation_probabilities = serializers.ListField(child=serializers.FloatField())
