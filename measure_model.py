import torch.nn as nn
from pathlib import Path
import torch
import csv
from torchvision import transforms
from PIL import Image
from pathlib import Path

class Conv_BoDiEs(nn.Module):
    def __init__(self):
        super(Conv_BoDiEs, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        # Calculate the flattened size
        # Input size is (b, 32, 12, 12) after conv4 and maxpool
        flattened_size = 32 * 12 * 12

        self.fc1 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def predict_and_save(model, model_path, image_path, result_file_path, device):
    """
    단일 이미지를 예측하고 결과를 CSV로 저장하는 함수.
    """
    # 측정 부위 리스트
    body_parts = [
        "chest circ", "waist circ", "pelvis circ", "neck circ", "bicep circ", 
        "thigh circ", "knee circ", "arm length", "leg length", "calf length", 
        "head circ", "wrist circ", "arm span", "shoulders width", "torso length", 
        "inner leg"
    ]

    # 이미지 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    # 단일 이미지 전처리
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('L')  # Grayscale 변환
        image = transform(image)
        image = image.unsqueeze(0)  # 배치 차원 추가
        return image

    # 이미지 전처리
    input_image = preprocess_image(image_path).to(device)

    # 모델 평가 모드 설정 및 예측 수행
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    # 예측 값 가져오기
    predicted_values = output.cpu().numpy()[0]  # 16개의 예측 값

    # 결과 출력
    print(f'Prediction for the image: {predicted_values}')

    # 결과 저장 경로 생성 (필요시)
    Path(result_file_path).parent.mkdir(parents=True, exist_ok=True)

    # CSV 파일로 결과 저장
    with open(result_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 첫 번째 행: 헤더
        writer.writerow(['Body Part', 'Prediction'])

        # 두 번째 행부터: 각 부위와 값
        for body_part, value in zip(body_parts, predicted_values):
            writer.writerow([body_part, value])

    print(f'Result saved to {result_file_path}')

# def predict_and_save(model, model_path, image_path, result_file_path, device):
#     """
#     단일 이미지를 예측하고 결과를 CSV로 저장하는 함수.
    
#     Parameters:
#     - model (nn.Module): 예측에 사용할 PyTorch 모델.
#     - image_path (str): 예측할 이미지 경로.
#     - result_file_path (str): 결과를 저장할 CSV 파일 경로.
#     - device (torch.device): 모델과 데이터를 실행할 디바이스 (CPU 또는 GPU).
#     """
#     # 이미지 전처리 정의
#     transform = transforms.Compose([
#         transforms.Resize((200, 200)),
#         transforms.ToTensor()
#     ])

#     # 단일 이미지 전처리
#     def preprocess_image(image_path):
#         image = Image.open(image_path).convert('L')  # Grayscale 변환
#         image = transform(image)
#         image = image.unsqueeze(0)  # 배치 차원 추가
#         return image

#     # 이미지 전처리
#     input_image = preprocess_image(image_path).to(device)

#     # 모델 평가 모드 설정 및 예측 수행
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     with torch.no_grad():
#         output = model(input_image)

#     # 예측 값 가져오기
#     predicted_value = output.cpu().numpy()[0]

#     # 결과 출력
#     print(f'Prediction for the image: {predicted_value}')

#     # 결과 저장 경로 생성 (필요시)
#     Path(result_file_path).parent.mkdir(parents=True, exist_ok=True)

#     # CSV 파일로 결과 저장
#     with open(result_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Body Part', 'Prediction'])
#         writer.writerow([Path(image_path).name, predicted_value])

#     print(f'Result saved to {result_file_path}')
