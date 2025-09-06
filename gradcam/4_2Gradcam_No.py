"""
Grad-CAM overlay & raw heat-map exporter
---------------------------------------
• 학습된 EfficientNet-B1 이진분류(yes/no) 모델을 로드
• 설정한 폴더 안 모든 이미지에 Grad-CAM 생성
• overlay PNG + raw .npy 저장
"""

import cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====================================
# 1. 경로 설정
# ====================================
device        = "cuda" if torch.cuda.is_available() else "cpu"
train_dir     = Path(r"C:\NEW_Aurora_jpg\Class_Random\pred\Pred_no")  ### ‼️‼️원본 이미지 경로 
weights_path  = "best_efficientnetb1_yesno.pth"   # fine-tuned 가중치
out_folder    = Path(r"C:\NEW_Aurora_jpg\Class_Random\pred\Pred_no\cam_overlays (Pred_no)0") ### ‼️‼️Grad cam 이미지 저장 경로 
out_folder.mkdir(exist_ok=True)

# ====================================
# 2. 학습된 모델 가중치 불러오기 + 출력기 교체 
# ====================================
model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
# head → 2-class
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device).eval()

# EfficientNetB1 최종 conv 블록
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ====================================
# 3. 전처리
# ====================================
to_tensor = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ====================================
# 4. Grad cam 계산
# ====================================
img_paths = sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
for img_path in img_paths:
    name = img_path.stem
    # 1) 원본 0-1 float RGB
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # 2) 텐서
    inp = to_tensor(Image.fromarray((rgb*255).astype(np.uint8))).unsqueeze(0).to(device)
    # 3) Grad-CAM (no=0) ‼️‼️no로 분류될 때 가장 큰 영향을 끼친 영역 
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(inp, targets=targets)[0]          # H×W 0-1
    orig_h, orig_w = rgb.shape[:2]
    grayscale_cam = cv2.resize(grayscale_cam, (orig_w, orig_h))
    # 5) overlay png이미지 시각화
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    cv2.imwrite(str(out_folder / f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Processed {name}")
    if device == "cuda":
        torch.cuda.empty_cache()

print(f"✅ 완료! overlay→ {out_folder}")
