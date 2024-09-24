from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    forehead_prediction = serializers.IntegerField()
    forehead_probabilities = serializers.ListField(child=serializers.FloatField())
    left_cheek_prediction = serializers.IntegerField()
    left_cheek_probabilities = serializers.ListField(child=serializers.FloatField())
    right_cheek_prediction = serializers.IntegerField()
    right_cheek_probabilities = serializers.ListField(child=serializers.FloatField())
