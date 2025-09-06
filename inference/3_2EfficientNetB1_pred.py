import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from PIL import Image
from pathlib import Path
import shutil
import numpy as np 

# ====================================
# 1. 경로 설정
# ====================================
source_folder = Path(r"C:\NEW_Aurora_jpg\Class\Astro\Moon_Full")  # 추론이 필요한 폴더 ‼️‼️원래는 Night + shutil.move였음 
pred_yes_folder = source_folder / "Pred_yes (Astro Full)"  # 예측 결과를 저장할 폴더
pred_no_folder  = source_folder / "Pred_no (Astro Full)"

pred_yes_folder.mkdir(parents=True, exist_ok=True)
pred_no_folder.mkdir(parents=True, exist_ok=True)

# 학습할 때 사용한 모델 가중치 파일 위치 
checkpoint_path = Path("best_efficientnetb1_yesno.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ====================================
# 2. 학습된 모델 가중치 불러오기 + 출력기 교체
# ====================================
# EfficientNet B1 아키텍처 + 가중치
weights = EfficientNet_B1_Weights.IMAGENET1K_V1
model = efficientnet_b1(weights=weights)

# 출력 레이어를 2개 클래스로 교체
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

# 가중치 로드
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# ====================================
# 3. 전처리/증강 (Inference 단계)
# ====================================
infer_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ====================================
# 4. 이진 분류 수행
# ====================================
# source_folder 최상위에 있는 JPG 파일만 처리 (train/val 등 하위 폴더 무시)
image_paths = sorted(source_folder.glob("*.jpg"))

with torch.no_grad():
    for img_path in image_paths:
        # 1) 이미지 열기
        img = Image.open(img_path).convert("RGB")
        inp = infer_transform(img).unsqueeze(0).to(device)  # (1,3,240,240)

        # 2) 예측
        outputs = model(inp)  # (1,2)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()  # (1,2)
        pred = np.argmax(probs)  # 0 or 1

        # 3) 결과에 따라 이동
        if pred == 1:
            # yes(오로라 있음)
            shutil.copy(str(img_path), str(pred_yes_folder / img_path.name))
        else:
            # no (오로라 없음)
            shutil.copy(str(img_path), str(pred_no_folder  / img_path.name))

        # 진행 상태 출력
        print(f"{img_path.name} -> {'yes' if pred==1 else 'no'}")

print("Inference complete. Total processed:", len(image_paths))
