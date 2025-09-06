from pathlib import Path
from skyfield.api import load, Topos
from skyfield.almanac import moon_phase
from datetime import datetime
import shutil

# ===== 경로 설정 =====
source_folder = Path(r"C:\NEW_Aurora_jpg\214890")
target_root = Path(r"C:\NEW_Aurora_jpg\Class")

# ===== 위치 설정 (옐로나이프) =====
latitude = 62.4333
longitude = -114.3500

# ===== Skyfield 설정 =====
ts = load.timescale()
sky = load('de421.bsp')
earth, sun, moon = sky['earth'], sky['sun'], sky['moon']
observer = earth + Topos(latitude_degrees=latitude, longitude_degrees=longitude)

# ===== 분류 함수 =====
def classify_sun_alt(alt):
    if alt <= -18:
        return "Night"
    elif -18 < alt <= -12:
        return "Astro"
    elif -12 < alt <= -6:
        return "Nautical"
    elif -6 < alt < 0:
        return "Civil"
    else:
        return "Day"

def classify_moon_phase(alt_deg, phase):
    if (phase >= 150)&(alt_deg > 5):
        return "Moon_Full"
    elif (45 <= phase < 150)&(alt_deg > 10):
        return "Moon_Quarter"
    else:
        return "NoMoon"

def extract_datetime(filename):
    try:
        timestamp = filename.split('_')[0]
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    except Exception as e:
        print(f"[오류] 시간 추출 실패: {filename} ({e})")
        return None

# ===== 메인 처리 =====
for img_path in source_folder.glob("*.jpg"):
    dt = extract_datetime(img_path.name)
    if dt is None:
        continue

    try:
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        obs = observer.at(t)

        # 🌞 태양 고도 → 시간대 분류
        sun_alt, _, _ = obs.observe(sun).apparent().altaz()
        sun_class = classify_sun_alt(sun_alt.degrees)

        # 🌙 달 고도 및 위상각 계산
        moon_alt, _, _ = obs.observe(moon).apparent().altaz()
        moon_phase_deg = 180 - abs(180 - moon_phase(sky, t).degrees)
        moon_class = classify_moon_phase(moon_alt.degrees, moon_phase_deg)

        # 폴더 경로 생성
        dest_folder = target_root / sun_class
        if moon_class != "NoMoon":
            dest_folder = dest_folder / moon_class
        dest_folder.mkdir(parents=True, exist_ok=True)

        # 파일 복사
        shutil.copy(img_path, dest_folder / img_path.name)
        print(f"[✅ 복사됨] {img_path.name} → {dest_folder}")

    except Exception as e:
        print(f"[⚠️ 오류] {img_path.name} 처리 중 문제 발생: {e}")
