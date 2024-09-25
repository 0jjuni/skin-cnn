
from langchain_community.llms import Ollama
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from accounts.models import PredictionResult
from rest_framework.permissions import IsAuthenticated
import traceback

# LLM 모델 초기화
llm = Ollama(model="gemma2")

# 나이, 색소 침착, 수분 상태, 스킨 타입에 대한 매핑 딕셔너리
AGE_PREDICTION_MAPPING = {
    0: "10대~20대",
    1: "30대~40대",
    2: "50대~60대"
}

PIGMENTATION_PREDICTION_MAPPING = {
    0: "색소침착이 없음",
    1: "색소침착이 있음"
}

MOISTURE_PREDICTION_MAPPING = {
    0: "수분이 없음",
    1: "수분이 있음"
}

SKIN_TYPE_PREDICTION_MAPPING = {
    0: "건성",
    1: "중성",
    2: "지성"
}

class GemmaAPIView(APIView):
    # 로그인된 사용자만 접근할 수 있도록 설정
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            # 로그인된 사용자의 예측 결과 가져오기
            user = request.user

            # 사용자의 가장 최근 PredictionResult 데이터 가져오기
            prediction_result = PredictionResult.objects.filter(user=user).latest('created_at')

            # 숫자를 설명으로 변환하여 포맷팅
            response_data = {
                'forehead_age_prediction': AGE_PREDICTION_MAPPING.get(prediction_result.forehead_age_prediction, '알 수 없음'),
                'left_cheek_age_prediction': AGE_PREDICTION_MAPPING.get(prediction_result.left_cheek_age_prediction, '알 수 없음'),
                'right_cheek_age_prediction': AGE_PREDICTION_MAPPING.get(prediction_result.right_cheek_age_prediction, '알 수 없음'),
                'forehead_pigmentation_prediction': PIGMENTATION_PREDICTION_MAPPING.get(prediction_result.forehead_pigmentation_prediction, '알 수 없음'),
                'forehead_moisture_prediction': MOISTURE_PREDICTION_MAPPING.get(prediction_result.forehead_moisture_prediction, '알 수 없음'),
                'left_cheek_moisture_prediction': MOISTURE_PREDICTION_MAPPING.get(prediction_result.left_cheek_moisture_prediction, '알 수 없음'),
                'right_cheek_moisture_prediction': MOISTURE_PREDICTION_MAPPING.get(prediction_result.right_cheek_moisture_prediction, '알 수 없음'),
                'forehead_skin_prediction': SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.forehead_skin_prediction, '알 수 없음'),
                'left_cheek_skin_prediction': SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.left_cheek_skin_prediction, '알 수 없음'),
                'right_cheek_skin_prediction': SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.right_cheek_skin_prediction, '알 수 없음')
            }

            # JSON 응답으로 반환
            return Response(response_data, status=status.HTTP_200_OK)

        except PredictionResult.DoesNotExist:
            return Response({'error': '예측 결과가 존재하지 않습니다.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return Response({'error': '알 수 없는 오류가 발생했습니다.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, *args, **kwargs):
        try:
            # 로그인된 사용자의 예측 결과 가져오기
            user = request.user

            # 사용자의 가장 최근 PredictionResult 데이터 가져오기
            prediction_result = PredictionResult.objects.filter(user=user).latest('created_at')

            formatted_data = (
                f"이마 나이 예측: {AGE_PREDICTION_MAPPING.get(prediction_result.forehead_age_prediction, '알 수 없음')}, "
                f"왼쪽 볼 나이 예측: {AGE_PREDICTION_MAPPING.get(prediction_result.left_cheek_age_prediction, '알 수 없음')}, "
                f"오른쪽 볼 나이 예측: {AGE_PREDICTION_MAPPING.get(prediction_result.right_cheek_age_prediction, '알 수 없음')}. "
                f"이마 색소 침착 예측: {PIGMENTATION_PREDICTION_MAPPING.get(prediction_result.forehead_pigmentation_prediction, '알 수 없음')}. "
                f"수분 상태: 이마 {MOISTURE_PREDICTION_MAPPING.get(prediction_result.forehead_moisture_prediction, '알 수 없음')}, "
                f"왼쪽 볼 {MOISTURE_PREDICTION_MAPPING.get(prediction_result.left_cheek_moisture_prediction, '알 수 없음')}, "
                f"오른쪽 볼 {MOISTURE_PREDICTION_MAPPING.get(prediction_result.right_cheek_moisture_prediction, '알 수 없음')}. "
                f"스킨 타입 예측: 이마 {SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.forehead_skin_prediction, '알 수 없음')}, "
                f"왼쪽 볼 {SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.left_cheek_skin_prediction, '알 수 없음')}, "
                f"오른쪽 볼 {SKIN_TYPE_PREDICTION_MAPPING.get(prediction_result.right_cheek_skin_prediction, '알 수 없음')}. "
                "피부 관리를 어떻게 하면 좋을까요?"
            )

            # LLM 모델에 질문 전송
            llm_response = llm.invoke(formatted_data)

            # JSON 응답으로 반환
            return Response({
                'formatted_data': formatted_data,
                'gemma_response': llm_response  # gemma 모델의 응답
            }, status=status.HTTP_200_OK)

        except PredictionResult.DoesNotExist:
            return Response({'error': '예측 결과가 존재하지 않습니다.'}, status=status.HTTP_404_NOT_FOUND)

            # 오류 핸들링 부분 수정
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            logging.error(traceback.format_exc())  # 전체 스택 트레이스 로그
            return Response({'error': f'알 수 없는 오류가 발생했습니다: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

