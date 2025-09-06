import random
import shutil
from pathlib import Path
from datetime import datetime

def sample_per_date(source_folder, target_folder):
    """
    날짜별로 최대 20장, 해당 날짜 이미지 수의 절반 이하(최소 1장)까지만 랜덤 샘플링하여
    target_folder에 옮김 
    """
    src = Path(source_folder)
    dst = Path(target_folder)
    dst.mkdir(parents=True, exist_ok=True)

    # 날짜별로 파일 목록을 모을 딕셔너리
    date_to_paths = {}

    # ===== source_folder 내 모든 .jpg 파일을 날짜(YYYY-MM-DD) 단위로 그룹화 =====
    for img_path in src.glob("*.jpg"):
        fname = img_path.name
        ts_str = fname[:14]  # "YYYYMMDDHHMMSS"
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        day_label = dt.strftime("%Y-%m-%d")
        date_to_paths.setdefault(day_label, []).append(img_path)

    # ===== 날짜별로 샘플링하여 target_folder에 복사 =====
    for day_label, paths in date_to_paths.items():
        n = len(paths)
        # 날짜별 최대 샘플 수 = min(20, floor(n/2)), 최소 1장
        max_allowed = max(n // 2, 1)
        sample_count = min(20, max_allowed)

        sampled = random.sample(paths, sample_count)
        for p in sampled:
            shutil.move(str(p), str(dst / p.name))

        print(f"{day_label}: 전체 {n}장 중 {sample_count}장 샘플링")

    print(f"\n총 샘플링된 이미지가 {len(list(dst.glob('*.jpg')))}장입니다.")


# ===== 실행 =====
source_folder = r"C:\NEW_Aurora_jpg\Class_Random"
target_folder = r"C:\NEW_Aurora_jpg\Class_Random\sample"
sample_per_date(source_folder, target_folder)
