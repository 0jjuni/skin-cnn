import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from PIL import Image
import torch.nn.functional as F
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import mediapipe as mp

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 첫 번째 모델: 나이 분류 모델
logging.info('나이 분류 모델 초기화 시작')
age_model = models.resnext50_32x4d(weights=None)
age_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(age_model.fc.in_features, 3)
)
age_model.load_state_dict(torch.load('./weights/나이_75.pth', map_location=torch.device('cpu')))
age_model.eval()

# 두 번째 모델: 색소 침착 모델
logging.info('색소 침착 모델 초기화 시작')
pigmentation_model = models.squeezenet1_0(weights=None)
pigmentation_model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)),
    nn.AdaptiveAvgPool2d((1, 1))
)
pigmentation_model.load_state_dict(torch.load('./weights/색소_78.pth', map_location=torch.device('cpu')))
pigmentation_model.eval()

# 세 번째 모델: 수분 예측 모델
logging.info('수분 예측 모델 초기화 시작')
moisture_model = models.resnext50_32x4d(weights=None)
moisture_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(moisture_model.fc.in_features, 2)  # 클래스 수 2개로 설정
)
moisture_model.load_state_dict(torch.load('./weights/수분_83.pth', map_location=torch.device('cpu')))
moisture_model.eval()

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
age_model.to(device)
pigmentation_model.to(device)
moisture_model.to(device)

# 이미지 전처리 정의
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_face_parts(image):
    """얼굴 부위 추출 (이마, 왼쪽 볼, 오른쪽 볼)"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # 이마 추출
        forehead_top = max(0, int(landmarks[10].y * h) - 30)
        forehead_bottom = min(h, int(landmarks[68].y * h))
        forehead_left = max(0, int(landmarks[21].x * w))
        forehead_right = min(w, int(landmarks[251].x * w))
        forehead = image[forehead_top:forehead_bottom, forehead_left:forehead_right]

        # 왼쪽 볼 추출
        left_cheek_left = int(landmarks[58].x * w)
        left_cheek_right = int(landmarks[203].x * w)
        left_cheek_top = int(landmarks[228].y * h)
        left_cheek_bottom = int(landmarks[214].y * h)
        left_cheek = image[left_cheek_top:left_cheek_bottom, left_cheek_left:left_cheek_right]

        # 오른쪽 볼 추출
        right_cheek_left = int(landmarks[423].x * w)
        right_cheek_right = int(landmarks[376].x * w)
        right_cheek_top = int(landmarks[448].y * h)
        right_cheek_bottom = int(landmarks[434].y * h)
        right_cheek = image[right_cheek_top:right_cheek_bottom, right_cheek_left:right_cheek_right]

        return forehead, left_cheek, right_cheek

    return None, None, None


def predict_age(image):
    """나이 예측"""
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = age_model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), probabilities.squeeze().cpu().numpy()

def predict_pigmentation(image):
    """색소 침착 예측"""
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = pigmentation_model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), probabilities.squeeze().cpu().numpy()

def predict_moisture(image):
    """수분 예측"""
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = moisture_model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), probabilities.squeeze().cpu().numpy()


class CropAndPredictAPIView(APIView):
    def post(self, request, *args, **kwargs):
        logging.info('API 요청 수신')

        if 'image' not in request.FILES:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # 얼굴 부위 추출 (이마, 왼쪽 볼, 오른쪽 볼)
        forehead, left_cheek, right_cheek = extract_face_parts(image)

        if forehead is not None and left_cheek is not None and right_cheek is not None:
            # 나이 예측
            predicted_age_class_forehead, age_probabilities_forehead = predict_age(forehead)
            predicted_age_class_left_cheek, age_probabilities_left_cheek = predict_age(left_cheek)
            predicted_age_class_right_cheek, age_probabilities_right_cheek = predict_age(right_cheek)

            # 색소 침착 예측 (이마만 대상으로)
            predicted_pigmentation_class, pigmentation_probabilities = predict_pigmentation(forehead)

            # 수분 예측 (이마, 왼쪽 볼, 오른쪽 볼 대상)
            predicted_moisture_class_forehead, moisture_probabilities_forehead = predict_moisture(forehead)
            predicted_moisture_class_left_cheek, moisture_probabilities_left_cheek = predict_moisture(left_cheek)
            predicted_moisture_class_right_cheek, moisture_probabilities_right_cheek = predict_moisture(right_cheek)

            # 예측 결과 반환
            return Response({
                # 나이 예측 결과
                '이마 나이 예측': predicted_age_class_forehead,
                '이마 나이 확률': f"{age_probabilities_forehead[predicted_age_class_forehead] * 100:.2f}%",  # 예측된 나이 클래스의 확률

                '왼쪽 볼 나이 예측': predicted_age_class_left_cheek,
                '왼쪽 볼 나이 확률': f"{age_probabilities_left_cheek[predicted_age_class_left_cheek] * 100:.2f}%",
                # 예측된 나이 클래스의 확률

                '오른쪽 볼 나이 예측': predicted_age_class_right_cheek,
                '오른쪽 볼 나이 확률': f"{age_probabilities_right_cheek[predicted_age_class_right_cheek] * 100:.2f}%",
                # 예측된 나이 클래스의 확률

                # 색소 침착 예측 결과 (이마만)
                '이마 색소 예측': predicted_pigmentation_class,
                '이마 색소 확률': f"{pigmentation_probabilities[predicted_pigmentation_class] * 100:.2f}%",  # 예측된 색소 클래스의 확률

                # 수분 예측 결과
                '이마 수분 예측': predicted_moisture_class_forehead,
                '이마 수분 확률': f"{moisture_probabilities_forehead[predicted_moisture_class_forehead] * 100:.2f}%",
                # 예측된 수분 클래스의 확률

                '왼쪽 볼 수분 예측': predicted_moisture_class_left_cheek,
                '왼쪽 볼 수분 확률': f"{moisture_probabilities_left_cheek[predicted_moisture_class_left_cheek] * 100:.2f}%",
                # 예측된 수분 클래스의 확률

                '오른쪽 볼 수분 예측': predicted_moisture_class_right_cheek,
                '오른쪽 볼 수분 확률': f"{moisture_probabilities_right_cheek[predicted_moisture_class_right_cheek] * 100:.2f}%",
                # 예측된 수분 클래스의 확률
            }, status=status.HTTP_200_OK)

        return Response({'error': 'Face not detected'}, status=status.HTTP_400_BAD_REQUEST)

