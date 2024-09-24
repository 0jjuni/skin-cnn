import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)  # INFO 레벨로 설정, 필요시 DEBUG로 변경 가능

# 기존에 사용하던 모듈들
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import cv2
import mediapipe as mp
import numpy as np

# Torch 관련 모듈들
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

# 1. 모델 정의 및 가중치 로드
logging.info('모델 초기화 시작')
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = models.resnext50_32x4d(weights=weights)
logging.info('모델 레이어 수정 완료')

# 출력 레이어 수정 (드롭아웃 비율 0.3)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 3)
)

# 가중치 로드
logging.info('모델 가중치 로드 시작')
model.load_state_dict(torch.load('./weights/나이_75.pth', map_location=torch.device('cpu')))
logging.info('모델 가중치 로드 완료')

# 평가 모드로 전환
logging.info('평가 모드로 전환 중')
model.eval()
logging.info('모델 평가 모드 완료')

# GPU 사용 가능하면 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"모델을 {'GPU' if torch.cuda.is_available() else 'CPU'}로 이동 완료")

# 이미지 전처리 정의
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
logging.info('이미지 전처리 설정 완료')


def predict_image(image):
    """이미지를 PIL 이미지로 변환 후 예측"""
    logging.info('이미지 예측 시작')
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    logging.info('이미지 예측 완료')
    return predicted.item(), probabilities.squeeze().cpu().numpy()


# MediaPipe Face Mesh 초기화
logging.info('MediaPipe Face Mesh 초기화 중')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
logging.info('MediaPipe Face Mesh 초기화 완료')


def extract_face_parts(image):
    logging.info('얼굴 부위 추출 시작')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        logging.info('얼굴 랜드마크 감지 완료')
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # 이마, 왼쪽 볼, 오른쪽 볼 추출
        forehead_top = max(0, int(landmarks[10].y * h) - 30)
        forehead_bottom = min(h, int(landmarks[68].y * h))
        forehead_left = max(0, int(landmarks[21].x * w))
        forehead_right = min(w, int(landmarks[251].x * w))
        forehead = image[forehead_top:forehead_bottom, forehead_left:forehead_right]

        left_cheek_left = int(landmarks[58].x * w)
        left_cheek_right = int(landmarks[203].x * w)
        left_cheek_top = int(landmarks[228].y * h)
        left_cheek_bottom = int(landmarks[214].y * h)
        left_cheek = image[left_cheek_top:left_cheek_bottom, left_cheek_left:left_cheek_right]

        right_cheek_left = int(landmarks[423].x * w)
        right_cheek_right = int(landmarks[376].x * w)
        right_cheek_top = int(landmarks[448].y * h)
        right_cheek_bottom = int(landmarks[434].y * h)
        right_cheek = image[right_cheek_top:right_cheek_bottom, right_cheek_left:right_cheek_right]

        logging.info('얼굴 부위 추출 완료')
        return forehead, left_cheek, right_cheek
    logging.warning('얼굴 랜드마크를 찾지 못함')
    return None, None, None


from .serializers import PredictionSerializer


class CropAndPredictAPIView(APIView):
    def post(self, request, *args, **kwargs):
        logging.info('API 요청 수신')
        if 'image' not in request.FILES:
            logging.warning('이미지 파일이 포함되지 않음')
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']

        # 이미지를 메모리에서 바로 처리
        logging.info('이미지 파일 처리 시작')
        image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        logging.info('이미지 파일 처리 완료')

        # 얼굴 부위 크롭
        logging.info('얼굴 부위 크롭 시작')
        forehead, left_cheek, right_cheek = extract_face_parts(image)

        if forehead is not None and left_cheek is not None and right_cheek is not None:
            # 딥러닝 모델에 크롭된 이미지 넣어 예측 수행
            logging.info('예측 시작: 이마')
            predicted_class_forehead, probabilities_forehead = predict_image(forehead)

            logging.info('예측 시작: 왼쪽 볼')
            predicted_class_left_cheek, probabilities_left_cheek = predict_image(left_cheek)

            logging.info('예측 시작: 오른쪽 볼')
            predicted_class_right_cheek, probabilities_right_cheek = predict_image(right_cheek)

            # 시리얼라이저로 예측 결과 포맷
            logging.info('예측 결과 포맷팅 시작')
            prediction_data = {
                'forehead_prediction': predicted_class_forehead,
                'forehead_probabilities': probabilities_forehead.tolist(),
                'left_cheek_prediction': predicted_class_left_cheek,
                'left_cheek_probabilities': probabilities_left_cheek.tolist(),
                'right_cheek_prediction': predicted_class_right_cheek,
                'right_cheek_probabilities': probabilities_right_cheek.tolist(),
            }

            serializer = PredictionSerializer(data=prediction_data)
            if serializer.is_valid():
                logging.info('예측 결과 반환 성공')
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                logging.error(f'시리얼라이저 에러: {serializer.errors}')
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        logging.warning('얼굴 감지 실패')
        return Response({'error': 'Face not detected'}, status=status.HTTP_400_BAD_REQUEST)
