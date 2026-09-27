"""청천일사량 계산과 기상청 단기예보 코드 변환.

[왜 필요한가]
기상청 단기예보(getVilageFcst)가 주는 14개 항목에 **일사량이 없다** — TMP(기온), WSD(풍속),
SKY(하늘상태), REH(습도), PTY, PCP, POP, SNO, TMN, TMX, UUU, VVV, VEC, WAV뿐이다.
그런데 태양광 컨버터는 `solar_rad`(MJ/m²)를 필수 입력으로 쓴다. 즉 예보만으로는 태양광 경로를
돌릴 수 없고, 일사량을 추정해야 한다.

추정의 골격은 물리다. 특정 위치·시각의 **청천일사량**(구름이 없을 때 지표에 도달하는 일사)은
태양 위치로 결정론적으로 계산된다. 실제 일사량은 거기에 구름이 곱하는 감쇠다.

    실제 일사량 ≈ 청천일사량 × 감쇠계수(하늘상태, 습도, ...)

감쇠계수는 ASOS 실측(일사량 + 전운량)으로 학습한다 — 이 저장소에 2021~2026년 자료가 있다
(app/training/train_radiation_model.py).

[또 하나의 정보 손실] 컨버터는 `cloud`를 전운량 0~10으로 받지만 예보는 SKY 코드(1/3/4)만 준다.
0~10을 3단계로 양자화하는 것이므로 정보가 줄고, 그만큼 성능이 떨어진다. 그 손실을
evaluate_forecast_degradation.py가 측정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ASOS 관측지점 좌표 (기상청 지점정보). 태양광 컨버터는 184(제주)를 쓴다.
STATIONS = {
    "184": {"name": "제주", "lat": 33.5141, "lon": 126.5297},
    "185": {"name": "고산", "lat": 33.2938, "lon": 126.1626},
    "188": {"name": "성산", "lat": 33.3862, "lon": 126.8802},
}

# 태양상수 W/m². 대기권 밖 수직면 일사량.
SOLAR_CONSTANT = 1367.0
# W/m² -> MJ/m²/h. 1 W/m² x 3600 s = 3600 J/m² = 0.0036 MJ/m².
W_TO_MJ_PER_HOUR = 0.0036

# 기상청 SKY 코드. 단기예보는 이 3단계만 준다(2는 쓰지 않는다).
SKY_LABELS = {1: "맑음", 3: "구름많음", 4: "흐림"}


def sky_from_cloud(cloud: np.ndarray | pd.Series) -> np.ndarray:
    """ASOS 전운량(0~10) -> 기상청 SKY 코드(1/3/4).

    기상청 기준: 운량 0~5 맑음(1), 6~8 구름많음(3), 9~10 흐림(4).
    ASOS를 예보와 같은 해상도로 떨어뜨려 학습·평가하기 위한 변환이다 — 이걸 하지 않으면
    전운량으로 학습한 모델을 SKY만 오는 예보에 넣게 되어 입력 분포가 어긋난다.
    """
    c = np.asarray(cloud, dtype=float)
    return np.where(np.isnan(c), np.nan, np.where(c <= 5, 1.0, np.where(c <= 8, 3.0, 4.0)))


def cloud_from_sky(sky: np.ndarray | pd.Series) -> np.ndarray:
    """SKY 코드 -> 대표 전운량. 각 구간의 중앙값을 쓴다(1->2.5, 3->7, 4->9.5).

    컨버터가 `cloud`를 0~10으로 받으므로 예보 입력에서는 역변환이 필요하다. 구간 대표값으로
    되돌리는 것이므로 **정보가 복원되지는 않는다** — 양자화 손실은 남는다.
    """
    s = np.asarray(sky, dtype=float)
    out = np.full(s.shape, np.nan)
    out[s == 1] = 2.5
    out[s == 3] = 7.0
    out[s == 4] = 9.5
    return out


def _solar_position(dt: pd.Series, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """태양 천정각의 코사인과 지구-태양 거리 보정계수.

    [시각 규약] 이 저장소는 `h시`를 '그 시간 구간의 끝'으로 쓴다(13시 = 12:00~13:00).
    일사량은 구간 적산값이므로 태양 위치는 **구간 중앙**(-30분)에서 계산해야 한다.
    이걸 빼먹으면 아침·저녁에 체계적으로 치우친다.
    """
    t = pd.to_datetime(dt) - pd.Timedelta(minutes=30)
    doy = t.dt.dayofyear.to_numpy(dtype=float)
    # 한국표준시(UTC+9) -> 진태양시. 경도 보정 + 시간차(equation of time).
    frac_hour = t.dt.hour.to_numpy(dtype=float) + t.dt.minute.to_numpy(dtype=float) / 60.0
    b = 2 * np.pi * (doy - 81) / 364.0
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = frac_hour + (lon - 135.0) * 4.0 / 60.0 + eot_min / 60.0  # 135°E = KST 기준자오선
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    latr = np.deg2rad(lat)
    cos_z = (np.sin(latr) * np.sin(decl)
             + np.cos(latr) * np.cos(decl) * np.cos(hour_angle))
    e0 = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)  # 이심률 보정
    return np.clip(cos_z, 0.0, None), e0


def clear_sky_mj(dt: pd.Series, station: str = "184") -> np.ndarray:
    """구름이 없을 때의 시간 적산 수평면 전일사량 (MJ/m²).

    Haurwitz 모델: GHI_clear = 1098 x cos(z) x exp(-0.059 / cos(z)) [W/m²].
    간단하지만 맑은 날 실측을 잘 감싸는 모델로 널리 쓰인다. 여기서는 절대값 정확도보다
    '시각·계절에 따른 상한 곡선'이 필요하므로 이 정도로 충분하다 — 감쇠계수를 ASOS로
    학습하면서 편향은 흡수된다.

    태양이 지평선 아래면 0이다.
    """
    st = STATIONS[station]
    cos_z, _ = _solar_position(dt, st["lat"], st["lon"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_w = np.where(cos_z > 0.01, 1098.0 * cos_z * np.exp(-0.059 / cos_z), 0.0)
    return ghi_w * W_TO_MJ_PER_HOUR


def extraterrestrial_mj(dt: pd.Series, station: str = "184") -> np.ndarray:
    """대기권 밖 수평면 일사량 (MJ/m²). 청천지수(kt)의 분모로 쓴다."""
    st = STATIONS[station]
    cos_z, e0 = _solar_position(dt, st["lat"], st["lon"])
    return SOLAR_CONSTANT * e0 * cos_z * W_TO_MJ_PER_HOUR


def add_solar_geometry(df: pd.DataFrame, station: str = "184") -> pd.DataFrame:
    """dt 열을 기준으로 청천일사량·대기권외 일사량·천정각 코사인을 추가한다."""
    st = STATIONS[station]
    out = df.copy()
    cos_z, _ = _solar_position(out["dt"], st["lat"], st["lon"])
    out["cos_zenith"] = cos_z
    out["clear_sky_mj"] = clear_sky_mj(out["dt"], station)
    out["extra_mj"] = extraterrestrial_mj(out["dt"], station)
    return out


# ---------------------------------------------------------------------------
# 기상청 격자 좌표 변환 (Lambert Conformal Conic)
# ---------------------------------------------------------------------------
# 단기예보는 전국을 5km x 5km 격자로 나눠 제공하고, 요청에 위경도가 아니라 격자번호(nx, ny)를
# 넣는다. 아래 상수는 기상청 '동네예보 격자 정보' 공개 파라미터다.
# 기준점 격자번호는 210/5 + 1 = 43, 675/5 + 1 = 136이다 — 격자번호가 1부터 시작하기 때문에
# +1이 붙는다. 이 +1을 빼먹으면 전국이 (1, 1)씩 밀린다(공식 예시 서울 종로구 60,127 / 제주시청
# 53,38로 검증했다).
_LCC = {"Re": 6371.00877, "grid": 5.0, "slat1": 30.0, "slat2": 60.0,
        "olon": 126.0, "olat": 38.0, "xo": 210 / 5.0 + 1, "yo": 675 / 5.0 + 1}


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 -> 기상청 단기예보 격자번호 (nx, ny).

    지점표를 찾아 옮겨 적는 대신 변환식을 두는 이유: 관측지점이 아니라 **발전단지 좌표**로
    예보를 받아야 하기 때문이다. 관측지점과 단지의 위치 불일치가 풍력 컨버터 오차의 원인이므로
    (README '컨버터'), 단지 좌표를 그대로 넣을 수 있어야 한다.
    """
    import math
    p = _LCC
    degrad = math.pi / 180.0
    re = p["Re"] / p["grid"]
    slat1, slat2 = p["slat1"] * degrad, p["slat2"] * degrad
    olon, olat = p["olon"] * degrad, p["olat"] * degrad
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    ra = math.tan(math.pi * 0.25 + lat * degrad * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * degrad - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    nx = int(ra * math.sin(theta) + p["xo"] + 0.5)
    ny = int(ro - ra * math.cos(theta) + p["yo"] + 0.5)
    return nx, ny
