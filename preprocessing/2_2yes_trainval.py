import random
import shutil
from pathlib import Path
from datetime import datetime

def split_yes_by_date(source_folder, train_folder, val_folder, train_ratio=0.8):
    """
    날짜별로 "yes 이미지 전부"를 train_ratio:1-train_ratio 비율로 나눠서
    train_folder와 val_folder로 이동
    """
    src = Path(source_folder)
    train_dst = Path(train_folder)
    val_dst = Path(val_folder)
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    # ===== source_folder 내 모든 .jpg 파일을 날짜(YYYY-MM-DD) 단위로 그룹화 =====
    date_to_paths = {} # 날짜별로 파일 목록을 모을 딕셔너리
    for img_path in src.glob("*.jpg"):
        fname = img_path.name
        ts_str = fname[:14]  # "YYYYMMDDHHMMSS"
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        day_label = dt.strftime("%Y-%m-%d")
        date_to_paths.setdefault(day_label, []).append(img_path)

    # ===== 날짜별로 80:20 비율로 train/val로 이동 =====
    for day_label, paths in date_to_paths.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        train_samples = paths[:n_train]
        val_samples = paths[n_train:]

        for p in train_samples:
            shutil.move(str(p), str(train_dst / p.name))
        for p in val_samples:
            shutil.move(str(p), str(val_dst / p.name))

        print(f"{day_label}: 전체 {n}장 → train {len(train_samples)}장, val {len(val_samples)}장")

    total_train = len(list(train_dst.glob("*.jpg")))
    total_val = len(list(val_dst.glob("*.jpg")))
    print(f"\n분할 완료: train(yes)={total_train}장, val(yes)={total_val}장")


# ===== 실행 예시 =====
source_yes = r"C:\NEW_Aurora_jpg\Class_Random\sample\yes"   # yes 이미지 폴더 (1,353장)
train_yes  = r"C:\NEW_Aurora_jpg\Class_Random\train\yes"    # train/yes 로 이동
val_yes    = r"C:\NEW_Aurora_jpg\Class_Random\val\yes"      # val/yes 로 이동

split_yes_by_date(source_yes, train_yes, val_yes, train_ratio=0.8)
