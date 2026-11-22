# features.py — 특징 생성 (도메인 그룹 소거 지원)
import math
from core import STATIONS, lg, fnum  # noqa
from core import _d
from weather import WXD  # noqa

# 도메인 그룹 정의 (소거 실험용)
G_WATER = ["IEM_WTRTP", "IEM_PH", "IEM_DOC", "IEM_TUR", "IEM_CHLA"]  # Chla는 lg로
G_BIO = ["IEM_BGALAGE_MICROSTS", "IEM_BGALAGE_ANBA", "IEM_BGALAGE_OSRTRIA", "IEM_BGALAGE_APZO"]
G_TASTE = ["IEM_GEOSM", "IEM_MIB2"]

WATER_LG = {"IEM_CHLA", "IEM_TUR"}
UPSTREAM = {"GJGR": "HP", "CS": "GJGR", "MGMR": "CS"}  # 낙동강 상류 체인
DOMAINS = ["hist", "water", "bio", "taste", "up", "lag"]


def lag_names():
    """시간 텀 도메인: 전·전전 실측 수질(lag1≈7일 전, lag2≈14일 전 — 주간 실측 케이던스) + lg_cell_t2"""
    out = ["lg_cell_t2"]
    for k in G_WATER:
        b = ("lg_" if k in WATER_LG else "") + k
        out += [b + "_lag1", b + "_lag2"]
    return out


def lag_row(past):
    """t1=전 실측(≈일주일 전), t2=전전 실측(≈14일 전) 수질 — WATER_LG는 lg 변형 동일 적용"""
    out = {"lg_cell_t2": lg(past[-3][1]) if len(past) >= 3 else None}
    for k in G_WATER:
        b = ("lg_" if k in WATER_LG else "") + k
        for idx, tag in ((-2, "lag1"), (-3, "lag2")):
            t = past[idx] if len(past) >= -idx else None
            v = t[2].get(k) if t else None
            out[b + "_" + tag] = lg(v) if (k in WATER_LG and v is not None and v >= 0) else (v if v is not None else None)
    return out


def domain_features(domains):
    """도메인 셋 → 특징 이름 리스트"""
    names = ["lg_cell_t0", "lg_cell_t1", "mon_sin", "mon_cos"]
    if "water" in domains:
        names += [f"{'lg_' if k in WATER_LG else ''}{k}" for k in G_WATER]
    if "bio" in domains:
        names += [f"lg_{k}" for k in G_BIO]
    if "taste" in domains:
        names += [f"lg_{k}" for k in G_TASTE]
    if "up" in domains:
        names += ["up_lg_cell", "up_missing"]
    if "lag" in domains:
        names += lag_names()
    if "wx" in domains:
        from weather import ANCHORS
        names += [f"wx_{a}_{n}" for a in ANCHORS for n in WXD]  # 95 확장: 앵커×기상(시간 텀 포함)
    return names


def build_xy(sid, series_sid, data, H, domains, val_years, test_years, fill=None, WX=None):
    """한 지점의 (X, y, meta) 생성. fill: {feat: value} 결측 대치(학습풀 통계 — 외부 주입 금지)
    t0=base 이전 마지막 실측(≤base), t1=그 이전"""
    from core import make_pairs
    pairs = make_pairs(series_sid, H, val_years, test_years)
    names = domain_features(domains)
    X, y, meta = [], [], []
    for p in pairs:
        base = p["base"]
        past = [(d, c, cv) for d, c, cv in series_sid if d <= base and c is not None]
        if not past:
            continue
        (d0, c0, cv0) = past[-1]
        (d1, c1, cv1) = past[-2] if len(past) > 1 else past[-1]
        row = {}
        row["lg_cell_t0"] = lg(c0)
        row["lg_cell_t1"] = lg(c1)
        row["mon_sin"] = math.sin(2 * math.pi * base.month / 12)
        row["mon_cos"] = math.cos(2 * math.pi * base.month / 12)
        if "water" in domains:
            for k in G_WATER:
                v = cv0.get(k)
                row[("lg_" if k in WATER_LG else "") + k] = lg(v) if (k in WATER_LG and v is not None and v >= 0) else (v if v is not None else None)
        if "bio" in domains:
            for k in G_BIO:
                v = cv0.get(k)
                row[f"lg_{k}"] = lg(v) if v is not None else None  # None→0-fill(부재=0 취급 아님, 대치)
        if "taste" in domains:
            for k in G_TASTE:
                v = cv0.get(k)
                row[f"lg_{k}"] = lg(v) if v is not None else None
        if "up" in domains:
            up = UPSTREAM.get(sid)
            if up:
                ups = data[up]
                upast = [(d, c) for d, c, _ in ups if d <= base and c is not None]
                if upast:
                    row["up_lg_cell"] = lg(upast[-1][1])
                    row["up_missing"] = 0.0
                else:
                    row["up_lg_cell"] = None
                    row["up_missing"] = 1.0
            else:
                row["up_lg_cell"] = None
                row["up_missing"] = 1.0
        if "lag" in domains:
            row.update(lag_row(past))
        if "wx" in domains and WX:
            from weather import wx_row_all
            row.update(wx_row_all(base, WX))
        vals = []
        ok = p["y"] is not None
        for n in names:
            v = row.get(n)
            if v is None:
                v = fill.get(n, 0.0) if fill else 0.0
            vals.append(v)
            if v is None:
                ok = False
        if not ok:
            continue
        X.append(vals)
        y.append(p["y"])
        meta.append({"sid": sid, "base": str(p["base"]), "target": str(p["target"]),
                     "pool": p["pool"], "y": p["y"]})
    return names, X, y, meta


def train_fill(data, sid, H, domains, val_years, test_years, WX=None):
    """학습 풀(train pairs)의 결측 대치값(특징별 중앙값) — 검증·테스트 정보 미사용"""
    from core import make_pairs
    names = domain_features(domains)
    pairs = make_pairs(data[sid], H, val_years, test_years)
    fill = {}
    med = {}
    dom_has_wx = "wx" in domains
    has_lag = "lag" in domains
    for p in pairs:
        if p["pool"] != "train":
            continue
        base = p["base"]
        past = [(d, c, cv) for d, c, cv in data[sid] if d <= base and c is not None]
        if not past:
            continue
        (d0, c0, cv0) = past[-1]
        if WX and dom_has_wx:
            from weather import wx_row_all
            for k, v in wx_row_all(base, WX).items():
                if v is not None:
                    med.setdefault(k, []).append(v)
        if has_lag:
            for k, v in lag_row(past).items():
                if v is not None:
                    med.setdefault(k, []).append(v)
        for k in G_WATER:
            v = cv0.get(k)
            if v is not None:
                med.setdefault(("lg_" if k in WATER_LG else "") + k, []).append(lg(v) if k in WATER_LG else v)
        for k in G_BIO + G_TASTE:
            v = cv0.get(k)
            if v is not None:
                med.setdefault(f"lg_{k}", []).append(lg(v))
    for n in names:
        lst = sorted(med.get(n, []))
        fill[n] = lst[len(lst) // 2] if lst else 0.0
    return fill


def truth_of(meta, pred_lg):
    """로그 예측을 원 단위로 환산해 (pred, y) 반환"""
    out_p, out_y = [], []
    for m, pv in zip(meta, pred_lg):
        out_p.append(max(10 ** pv - 1, 0))
        out_y.append(m["y"])
    return out_p, out_y
