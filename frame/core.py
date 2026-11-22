# AIDA HAB backtest frame — stdlib only (numpy/pandas 불안정 환경 회피)
# core.py: 데이터 적재 + 호라이즌 pair 생성 + 지표
import json, math, os
from datetime import date, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ["HP", "GJGR", "CS", "MGMR", "DCHN"]
SDN = {"HP": "해평", "GJGR": "강정·고령", "CS": "칠서", "MGMR": "물금·매리", "DCHN": "회남"}
CELL = "IEM_BGALAGE_CELL_CO"
COVS = ["IEM_WTRTP", "IEM_PH", "IEM_DOC", "IEM_TRP", "IEM_TUR", "IEM_CHLA",
        "IEM_BGALAGE_MICROSTS", "IEM_BGALAGE_ANBA", "IEM_BGALAGE_OSRTRIA", "IEM_BGALAGE_APZO",
        "IEM_GEOSM", "IEM_MIB2",
        "IEM_MICROSTLR", "IEM_MICROSTRR", "IEM_MICROSTLA", "IEM_MICROSTYR",
        "IEM_MICROSTLF", "IEM_MICROSTLY"]


def _d(s):
    y, m, dd = s.split(".")
    return date(int(y), int(m), int(dd))


def fnum(x):
    if x in (None, "", "NULL", "-"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_master():
    """master json → {sid: [(d: date, cell: float|None, covs: dict)]} 날짜 오름차순"""
    p = os.path.join(BASE, os.pardir, "data", "algae_measure_master_2016_2026.json")
    rows = json.load(open(p, encoding="utf8"))
    out = {s: [] for s in STATIONS}
    for r in rows:
        sid = r["SID"]
        covs = {k: fnum(r.get(k)) for k in COVS}
        out[sid].append((_d(r["CHCK_DE"]), fnum(r.get(CELL)), covs))
    for s in out:
        out[s].sort(key=lambda t: t[0])
    return out


def make_pairs(series, horizon, val_years=(2024,), test_years=(2025, 2026)):
    """누출 방지 호라이즌 pair: base=실측일, target=base+H 이후 최근접 실측일(±3일)
    학습: target의 연도가 val/test가 아닌 전체 / 검증: val_years / 테스트: test_years
    리턴: [(sid, base_d, target_d, target_cell)] — 지점별 독립"""
    dates = [t[0] for t in series]
    pos = {d: i for i, d in enumerate(dates)}
    pairs = []
    tgt_y = val_years + test_years
    for i, (d0, _, _) in enumerate(series):
        want = d0 + timedelta(days=horizon)
        # target: want 이후 최근접 실측일, want-3 이상이어야 (±3일 매칭)
        j = i
        while j < len(dates) and dates[j] < want:
            j += 1
        if j >= len(dates) or dates[j] - want > timedelta(days=3):
            continue
        ty, cell = series[j][0].year, series[j][1]
        pool = "val" if ty in val_years else ("test" if ty in test_years else "train")
        pairs.append({"sid": None, "base": d0, "target": dates[j], "y": cell, "pool": pool})
    return pairs


# ---- 지표 ----
def rmse(pred, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, y)) / len(pred))


def mae(pred, y):
    return sum(abs(a - b) for a, b in zip(pred, y)) / len(pred)


def pearson_r2(pred, y):
    n = len(pred)
    if n < 3:
        return None
    mp = sum(pred) / n
    my = sum(y) / n
    sxy = sum((a - mp) * (b - my) for a, b in zip(pred, y))
    sx = math.sqrt(sum((a - mp) ** 2 for a in pred))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return (sxy / sx / sy) ** 2


def nse(pred, y):
    n = len(pred)
    my = sum(y) / n
    den = sum((b - my) ** 2 for b in y)
    if den == 0:
        return None
    num = sum((a - b) ** 2 for a, b in zip(pred, y))
    return 1 - num / den


def f1_event(pred, y, thr=10000.0):
    """세포수≥thr(10,000 cells/mL) 이벤트 탐지 F1"""
    tp = fp = fn = 0
    for a, b in zip(pred, y):
        p, t = a >= thr, b >= thr
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif t and not p:
            fn += 1
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def lg(x):
    return math.log10(x + 1)


def eval_all(pred, y):
    """pred/y: 실제 세포수(원 단위) → 로그공간 지표 + 원단위 R²/NSE + F1"""
    lp = [lg(max(x, 0)) for x in pred]
    ly = [lg(max(b, 0)) for b in y]
    return {"rmse_log": rmse(lp, ly), "mae_log": mae(lp, ly),
            "r2_raw": pearson_r2(pred, y), "nse": nse(pred, y),
            "f1_10k": f1_event(pred, y)}
