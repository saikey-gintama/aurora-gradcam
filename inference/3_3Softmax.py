import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from PIL import Image
from pathlib import Path
import numpy as np

# ===============================
# 1. 설정
# ===============================
root_folder = Path(r"C:\NEW_Aurora_jpg\Class_Random\train")
checkpoint_path = Path("best_efficientnetb1_yesno.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ===============================
# 2. 모델 로딩
# ===============================
weights = EfficientNet_B1_Weights.IMAGENET1K_V1
model = efficientnet_b1(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# ===============================
# 3. 전처리
# ===============================
infer_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ===============================
# 4. 모든 하위 폴더 순회하며 softmax 저장 + 통계
# ===============================
subfolders = [f for f in root_folder.iterdir() if f.is_dir()]
print("총 폴더 수:", len(subfolders))
batch_size = 64  # GPU 메모리에 따라 조절 (8, 16, 32, 64 등으로 테스트 권장)

for folder in subfolders:
    image_paths = sorted(folder.glob("*.jpg"))
    output_txt = folder / f"softmax_probs_{folder.name}.txt"
    if len(image_paths) == 0:
        print(f"[{folder.name}] 이미지 없음. 스킵.")
        continue

    no_probs = []
    yes_probs = []

    with open(output_txt, "w") as f:
        f.write("filename,no_prob,yes_prob\n")
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]

                imgs = [infer_transform(Image.open(p).convert("RGB")) for p in batch_paths]
                inp = torch.stack(imgs).to(device)

                outputs = model(inp)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for img_path, prob in zip(batch_paths, probs):
                    f.write(f"{img_path.name},{prob[0]:.6f},{prob[1]:.6f}\n")
                    no_probs.append(prob[0])
                    yes_probs.append(prob[1])

        # 기본 통계 계산
        no_stats = np.array(no_probs)
        f.write(
            "[STATS_no]," +
            f"mean={no_stats.mean():.6f}," +
            f"std={no_stats.std():.6f}," +
            f"min={no_stats.min():.6f}," +
            f"max={no_stats.max():.6f}," +
            f"median={np.median(no_stats):.6f}\n")
        
        yes_stats = np.array(yes_probs)
        f.write(
            "[STATS_yes]," +
            f"mean={yes_stats.mean():.6f}," +
            f"std={yes_stats.std():.6f}," +
            f"min={yes_stats.min():.6f}," +
            f"max={yes_stats.max():.6f}," +
            f"median={np.median(yes_stats):.6f}\n")

    print(f"[{folder.name}] Softmax 저장 완료 ({len(image_paths)}장): {output_txt}")