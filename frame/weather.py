# weather.py — ERA5-Land 일자료 로더 + wx 특징 계산 (Open-Meteo archive, 무인증)
# 소스: data/weather_raw/era5land_<anchor>.json (2014-01-01 → 2026-11-17, 4,704일 무결)
# 앵커 = 01_과제_스터디.md §1 근접 ASOS 매핑: HP→구미, GJGR→대구, CS→의령, MGMR→부산, DCHN→대전(대청호)
import json, os
from datetime import date, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
WR = os.path.join(BASE, os.pardir, "data", "weather_raw")

ANCHOR_OF = {"HP": "gumi", "GJGR": "daegu", "CS": "uiryeong", "MGMR": "busan", "DCHN": "daejeon",
             "IC": "icheon", "MJ": "daejeon", "CJ": "cheongju"}  # v5 8국 확장(2026-11-19)
# 시간 텀 확장(2026-11-18): l7 단일 지연(일주일 전), r14/r30 장기 창, 일조시간 신규 — 19 × 5앵커 = 95
WXD = ("prcp_l1", "prcp_r3", "prcp_r7", "prcp_r14", "prcp_r30",
       "tmax_l1", "tmax_l7", "tmax_r7", "tmax_r14",
       "tmin_r7", "tmin_r14", "rh_r7", "rh_r14", "ws_l1", "ws_r7",
       "sun_l1", "sun_l7", "sun_r7", "sun_r14")
ANCHORS = ["gumi", "daegu", "uiryeong", "busan", "daejeon"]
ANCHORS7 = ANCHORS + ["icheon", "cheongju"]  # v5 실험 C 전용(7앵커·133 wx)


def load_weather(anchors=None):
    """{anchor: {date: {prcp,tmax,tmin,ws(m/s),rh,sun(시간)}}} — 일 단위 완전 커버 전제
    강우·풍속·일조는 ERA5 archive 기본 모델(era5_prcp_*/era5_ws_*/era5_sun_*.json) 병합
    anchors=None → 기존 5앵커(기존 실험 A7/A/B 불변), 실험 C는 weather.ANCHORS7 전달"""
    out = {}
    for a in (anchors or ANCHORS):
        d = json.load(open(os.path.join(WR, f"era5land_{a}.json"), encoding="utf8"))["daily"]
        prcp_map = {}
        ws_map = {}
        sun_map = {}
        pp = os.path.join(WR, f"era5_prcp_{a}.json")
        if os.path.exists(pp):
            pd = json.load(open(pp, encoding="utf8"))["daily"]
            for i, ds in enumerate(pd["time"]):
                y, m, dd = ds.split("-")
                prcp_map[date(int(y), int(m), int(dd))] = pd["precipitation_sum"][i]
        wsfp = os.path.join(WR, f"era5_ws_{a}.json")
        if os.path.exists(wsfp):
            wd = json.load(open(wsfp, encoding="utf8"))["daily"]
            for i, ds in enumerate(wd["time"]):
                y, m, dd = ds.split("-")
                wv = wd["wind_speed_10m_max"][i]
                ws_map[date(int(y), int(m), int(dd))] = (wv / 3.6) if wv is not None else None
        sfp = os.path.join(WR, f"era5_sun_{a}.json")
        if os.path.exists(sfp):
            sd = json.load(open(sfp, encoding="utf8"))["daily"]
            for i, ds in enumerate(sd["time"]):
                y, m, dd = ds.split("-")
                sv = sd["sunshine_duration"][i]
                sun_map[date(int(y), int(m), int(dd))] = (sv / 3600.0) if sv is not None else None
        rows = {}
        for i, ds in enumerate(d["time"]):
            y, m, dd = ds.split("-")
            bd = date(int(y), int(m), int(dd))
            rows[bd] = {
                "prcp": prcp_map.get(bd, d["precipitation_sum"][i]),
                "tmax": d["temperature_2m_max"][i],
                "tmin": d["temperature_2m_min"][i],
                "ws": ws_map.get(bd, (d["wind_speed_10m_max"][i] / 3.6) if d["wind_speed_10m_max"][i] is not None else None),
                "rh": d["relative_humidity_2m_mean"][i],
                "sun": sun_map.get(bd),
            }
        out[a] = rows
    return out


def wx_row(anchor, base, WX):
    """base 기준 base-1..base-14 기상 → wx_<name> 특징 dict (누락 시 구성 특징 None→대치)
    강우: 지연합(전일~14일 누적 antecedent), 온도·습도: 7일 평균, 풍속: 전일 최대(10m)"""
    w = WX.get(anchor) if WX else None
    if not w:
        return None
    def day(k):
        return w.get(base - timedelta(days=k))
    def agg(k1, k2, key, mode):
        vals = []
        for k in range(k1, k2 + 1):
            r = day(k)
            if r and r.get(key) is not None:
                vals.append(r[key])
        if not vals:
            return None
        return (sum(vals) / len(vals)) if mode == "mean" else sum(vals)
    d1 = day(1)
    out = {
        "wx_prcp_l1": None if not d1 else d1.get("prcp"),
        "wx_prcp_r3": agg(1, 3, "prcp", "sum"),
        "wx_prcp_r7": agg(1, 7, "prcp", "sum"),
        "wx_prcp_r14": agg(1, 14, "prcp", "sum"),
        "wx_prcp_r30": agg(1, 30, "prcp", "sum"),
        "wx_tmax_l1": None if not d1 else d1.get("tmax"),
        "wx_tmax_l7": None if not day(7) else day(7).get("tmax"),
        "wx_tmax_r7": agg(1, 7, "tmax", "mean"),
        "wx_tmax_r14": agg(1, 14, "tmax", "mean"),
        "wx_tmin_r7": agg(1, 7, "tmin", "mean"),
        "wx_tmin_r14": agg(1, 14, "tmin", "mean"),
        "wx_rh_r7": agg(1, 7, "rh", "mean"),
        "wx_rh_r14": agg(1, 14, "rh", "mean"),
        "wx_ws_l1": None if not d1 else d1.get("ws"),
        "wx_ws_r7": agg(1, 7, "ws", "mean"),
        "wx_sun_l1": None if not d1 else d1.get("sun"),
        "wx_sun_l7": None if not day(7) else day(7).get("sun"),
        "wx_sun_r7": agg(1, 7, "sun", "sum"),
        "wx_sun_r14": agg(1, 14, "sun", "sum"),
    }
    return out


def wx_row_all(base, WX, anchors=None):
    """95 확장: (앵커 목록 기본=WX 실제 키 삽입 순서 — 기존 5앵커는 ANCHORS 순서 동일) × 19 → wx_<anchor>_<name>
    C 실험은 load_weather(ANCHORS7) 결과(삽입 순서 = ANCHORS7 순서) 전달"""
    out = {}
    for a in (anchors if anchors is not None else list(WX.keys())):
        r = wx_row(a, base, WX)
        if r:
            for k, v in r.items():
                out[f"wx_{a}_{k[3:]}"] = v
    return out
