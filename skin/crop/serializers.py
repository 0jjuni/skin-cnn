from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    # 나이 예측 결과
    forehead_age_prediction = serializers.IntegerField()
    forehead_age_probabilities = serializers.ListField(child=serializers.FloatField())

    left_cheek_age_prediction = serializers.IntegerField()
    left_cheek_age_probabilities = serializers.ListField(child=serializers.FloatField())

    right_cheek_age_prediction = serializers.IntegerField()
    right_cheek_age_probabilities = serializers.ListField(child=serializers.FloatField())

    # 색소 침착 예측 결과
    forehead_pigmentation_prediction = serializers.IntegerField()
    forehead_pigmentation_probabilities = serializers.ListField(child=serializers.FloatField())

    # 수분 예측 결과
    forehead_moisture_prediction = serializers.IntegerField()
    forehead_moisture_probabilities = serializers.ListField(child=serializers.FloatField())

    left_cheek_moisture_prediction = serializers.IntegerField()
    left_cheek_moisture_probabilities = serializers.ListField(child=serializers.FloatField())

    right_cheek_moisture_prediction = serializers.IntegerField()
    right_cheek_moisture_probabilities = serializers.ListField(child=serializers.FloatField())

    # # 스킨 타입 예측 결과
    # forehead_skin_type_prediction = serializers.IntegerField()
    # forehead_skin_type_probabilities = serializers.ListField(child=serializers.FloatField())

