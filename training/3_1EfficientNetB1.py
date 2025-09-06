import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torch.utils.data import DataLoader
from pathlib import Path

def train_and_validate():
    # ====================================
    # 1. 경로 및 하이퍼파라미터 설정
    # ====================================
    train_dir    = Path(r"C:\NEW_Aurora_jpg\Class_Random\train")  # train할 이미지를 저장할 폴더 
    val_dir      = Path(r"C:\NEW_Aurora_jpg\Class_Random\val")    # val할 이미지를 저장할 폴더 

    batch_size    = 16       # 2070 Max-Q 8GB VRAM에서 안정적
    num_epochs    = 12       # 6 에포크 freeze + 6 에포크 fine-tune
    learning_rate = 3e-4
    num_classes   = 2        # yes / no
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ====================================
    # 2. 데이터 증강 & 데이터로더
    # ====================================
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(240, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset   = ImageFolder(val_dir,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ====================================
    # 3. 모델 불러오기 + 분류기 교체
    # ====================================
    # EfficientNet B1 사전학습 가중치를 weights 인자로 지정
    weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    model = efficientnet_b1(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # ====================================
    # 4. 손실 함수, 옵티마이저
    # ====================================
    criterion = nn.CrossEntropyLoss()

    # 처음 6 에포크는 classifier만 학습 (backbone freeze)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

    # ====================================
    # 5. Training & Validation 루프
    # ====================================
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ------- Train 단계 -------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = running_corrects / len(train_dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # ------- Validation 단계 -------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += (preds == labels).sum().item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc  = val_corrects / len(val_dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Val   Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

        # 베스트 모델 저장
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_efficientnetb1_yesno.pth")

        # ------- Fine-tuning (Epoch 6 이후) -------
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)
            print("→ Fine-tuning: backbone도 학습 가능하도록 변경, LR을 10배 낮춤")

    print(f"최종 베스트 Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_and_validate()
