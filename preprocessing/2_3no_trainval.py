import random
import shutil
from pathlib import Path
from datetime import datetime
import math

def split_no_by_date_exact(source_folder, train_folder, val_folder,
                           total_train_yes, total_val_yes, total_no):
    """
    날짜별로 'no' 이미지를 정확히 total_train_yes, total_val_yes 비율에 맞춰
    train_folder와 val_folder로 이동
    """
    train_folder = Path(train_folder); train_folder.mkdir(parents=True, exist_ok=True)
    val_folder   = Path(val_folder);   val_folder.mkdir(parents=True, exist_ok=True)
    src = Path(source_folder)

    # ===== source_folder 내 모든 .jpg 파일을 날짜(YYYY-MM-DD) 단위로 그룹화 =====
    date_to_paths = {} # 날짜별로 파일 목록을 모을 딕셔너리
    for img_path in src.glob("*.jpg"):
        ts_str = img_path.name[:14]  # "YYYYMMDDHHMMSS"
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        day = dt.strftime("%Y-%m-%d")
        date_to_paths.setdefault(day, []).append(img_path)

    dates = sorted(date_to_paths.keys())  # 각 날짜별 이미지 수 n_i
    counts = {day: len(date_to_paths[day]) for day in dates}

    # ===== 날짜별 train 비율 적용 =====
    train_ratio = total_train_yes / total_no # 전체 no 중 몇 %를 train에 넣을지
    train_floats = {}
    for day in dates:
        train_floats[day] = counts[day] * train_ratio

    train_alloc = {day: math.floor(train_floats[day]) for day in dates} # 정수로 맞추기

    # 소수점 큰 날짜부터 하나씩 더 이동
    train_fraction = {day: train_floats[day] - train_alloc[day] for day in dates} # 소수점은 모아모아 
    allocated = sum(train_alloc.values())
    remainder = total_train_yes - allocated
    for day in sorted(dates, key=lambda d: train_fraction[d], reverse=True)[:remainder]:
        train_alloc[day] += 1

    # 날짜별로 train_i만큼 랜덤 이동
    leftover = {}
    for day in dates:
        paths = date_to_paths[day]
        random.shuffle(paths)
        n_train_i = train_alloc[day]
        train_paths = paths[:n_train_i]
        for p in train_paths:
            shutil.move(str(p), str(train_folder / p.name))
        # leftover에 남은 이미지 (val 계산에 적용됨)
        leftover[day] = paths[n_train_i:]

    leftover_sum = sum(len(leftover[day]) for day in dates)

    # ===== 날짜별 val 비율 적용 =====
    val_ratio = total_val_yes / leftover_sum # leftover 중에서 몇 %를 val로 사용할지
    val_floats = {}
    for day in dates:
        val_floats[day] = len(leftover[day]) * val_ratio

    val_alloc = {day: math.floor(val_floats[day]) for day in dates} # 정수로 맞추기

    # 소수점 큰 날짜부터 하나씩 더 이동
    val_fraction = {day: val_floats[day] - val_alloc[day] for day in dates} # 소수점은 모아모아 
    allocated_val = sum(val_alloc.values())
    remainder_val = total_val_yes - allocated_val
    for day in sorted(dates, key=lambda d: val_fraction[d], reverse=True)[:remainder_val]:
        val_alloc[day] += 1

    # 날짜별로 val_i만큼 랜덤 이동
    for day in dates:
        paths = leftover[day]
        random.shuffle(paths)
        n_val_i = val_alloc[day]
        val_paths = paths[:n_val_i]
        for p in val_paths:
            shutil.move(str(p), str(val_folder / p.name))

    # ===== 출력 =====
    moved_train = sum(train_alloc.values())
    moved_val   = sum(val_alloc.values())
    print(f"총 이동된 no 이미지 → train_no: {moved_train}장, val_no: {moved_val}장")


# ===== 실행 =====
source_no = r"C:\NEW_Aurora_jpg\Class_Random\sample\no"     # no 이미지 폴더 (총 3,210장)
train_no  = r"C:\NEW_Aurora_jpg\Class_Random\train\no"      # train/no 로 이동
val_no    = r"C:\NEW_Aurora_jpg\Class_Random\val\no"        # val/no 로 이동

total_train_yes = 1024
total_val_yes   = 329
total_no_images = 3210

split_no_by_date_exact(source_no, train_no, val_no,
                       total_train_yes, total_val_yes, total_no_images)