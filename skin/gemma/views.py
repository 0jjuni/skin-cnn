import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from accounts.models import PredictionResult
from .serializers import PredictionResultSerializer
from langchain_community.llms import Ollama

# LLM 모델 초기화
llm = Ollama(model="gemma2")

class GemmaAPIView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # 로그인된 사용자의 예측 결과 가져오기
            user = request.user
            if not user.is_authenticated:
                return Response({'error': '로그인이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)

            # 사용자의 가장 최근 PredictionResult 데이터 가져오기
            prediction_result = PredictionResult.objects.filter(user=user).latest('created_at')
            serializer = PredictionResultSerializer(prediction_result)

            # 가져온 데이터 포맷팅
            formatted_data = (
                f"이마 나이 예측은 {serializer.data['forehead_age_prediction']}이고, "
                f"확률은 {serializer.data['forehead_age_probability']}입니다. "
                f"왼쪽 볼 나이 예측은 {serializer.data['left_cheek_age_prediction']}이고, "
                f"확률은 {serializer.data['left_cheek_age_probability']}입니다. "
                f"오른쪽 볼 나이 예측은 {serializer.data['right_cheek_age_prediction']}이고, "
                f"확률은 {serializer.data['right_cheek_age_probability']}입니다. "
                f"이마의 색소 예측은 {serializer.data['forehead_pigmentation_prediction']}이며, "
                f"확률은 {serializer.data['forehead_pigmentation_probability']}입니다. "
                f"수분 상태는 이마에서 {serializer.data['forehead_moisture_prediction']}이고, "
                f"확률은 {serializer.data['forehead_moisture_probability']}입니다. "
                "피부 관리를 어떻게 하면 좋을까요?"
            )

            # LLM 모델에 질문 전송
            llm_response = llm.invoke(formatted_data)

            # JSON 응답으로 반환
            return Response({
                'formatted_data': formatted_data,
                'gemma_response': llm_response['text']  # gemma 모델의 응답
            }, status=status.HTTP_200_OK)

        except PredictionResult.DoesNotExist:
            return Response({'error': '예측 결과가 존재하지 않습니다.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return Response({'error': '알 수 없는 오류가 발생했습니다.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
