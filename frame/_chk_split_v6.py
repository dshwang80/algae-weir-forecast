# _chk_split_v6.py — ③ 데이터 분할 경계 감사 (11-22 유저 지시: 투고 전 확인)
# 핵심: make_pairs pool = target 연도 기준 (core.py L62) → 목표값 경계 유출 구조적으로 원천차단 검증
import csv, os, sys
from collections import Counter, defaultdict
import core, run_v6

data, n_rows = run_v6.load_16()

def audit(H):
    recs = []
    for s in run_v6.STATIONS16:
        for p in core.make_pairs(data[s], H, val_years=(2024,), test_years=(2025, 2026)):
            p["sid"] = s
            recs.append(p)
    full = [p for p in recs if p["y"] is not None]          # Full arm (build_xy와 동일 필터)
    bad = []   # target-year 경계침범
    dt = defaultdict(list)
    for p in full:
        ty = p["target"].year
        exp = "val" if ty == 2024 else ("test" if ty in (2025, 2026) else "train")
        if p["pool"] != exp:
            bad.append((p["sid"], str(p["target"]), p["pool"], exp))
        dt[p["pool"]].append((p["target"] - p["base"]).days)
    cnt = Counter(p["pool"] for p in full)
    raw = Counter(p["pool"] for p in recs)
    def mm(pool):
        ds = dt[pool]
        return (min(ds), max(ds)) if ds else (None, None)
    out = dict(H=H, raw=dict(raw), full=dict(cnt), target_leak=bad,
               lead=mm("train"), lead_val=mm("val"), lead_test=mm("test"))
    # H=3 상한: target−base==3 (3일 확인지연 가정 시 발행일=목표일 가능한 최대 부분집합)
    if H == 3:
        co = {"val": sum(1 for p in full if p["pool"] == "val" and (p["target"] - p["base"]).days == 3),
              "test": sum(1 for p in full if p["pool"] == "test" and (p["target"] - p["base"]).days == 3),
              "train": sum(1 for p in full if p["pool"] == "train" and (p["target"] - p["base"]).days == 3)}
        out["h3_coincide_ub"] = co
    return out

print("rows loaded:", n_rows, "stations:", len(run_v6.STATIONS16))
for H in (3, 7, 14):
    o = audit(H)
    print("\nH=%d  raw pairs=%s  y-present=%s" % (H, o["raw"], o["full"]))
    print(" leak(≠target-year pool)= %d" % len(o["target_leak"]))
    print(" train target−base: %d..%d   val: %s..%s   test: %s..%s" % (
        o["lead"][0], o["lead"][1], o["lead_val"][0], o["lead_val"][1], o["lead_test"][0], o["lead_test"][1]))
    if "h3_coincide_ub" in o:
        print(" H=3 target−base==3 (동일일 발행 상한):", o["h3_coincide_ub"])
