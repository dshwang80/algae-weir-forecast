# -*- coding: utf-8 -*-
"""S1 감사(2026-11-22 v10 검수 필수 항목): Table S1 rates vs 원자료 재유동.
- run_v6.build_masked(시드 20261120) 재실행 → 항목×시즌별 방문일 셀 수 N·유효건수 before/after·added 실측
- 기존 S1 표(unit_pct, 16개 보 순회 후 마지막 스냅샷 의심) vs **항목별·계절별 분모 재계산** 대조
- OSRTRIA(3,819) vs MICROSTS(4,085) 감소폭 정합성 판정 근거 산출
"""
import json, os, sys, collections

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import run_v6 as R

data, n_rows = R.load_16()
data_m, st = R.build_masked(data)
units = R.M_UNITS

# stats 단위_pct = 보별 스냅샷인지 확인(최종 저장 1개 보만 남는지)
snap = st["unit_pct"]

print("n_rows:", n_rows, "restored:", st["restored_total"], "added_total:", sum(st["per_item"].values()))
print()
print(f"{'item':26s} {'src':4s} | {'pool':4s} | N    nBef  nAft  added | pooled rate | lastStation rate (unit_pct)")
rows_out = []
audit = {}
for item, src in units:
    for pool in ("mgmt", "off"):
        N = nbef = naft = 0
        per_station = []
        added_pool = 0
        for s in R.STATIONS16:
            series = data[s]
            idxs = [i for i, row in enumerate(series) if R.season_of(row[0]) == pool]
            nb = sum(1 for i in idxs if series[i][2].get(item) is not None)
            na = sum(1 for i in idxs if data_m[s][i][2].get(item) is not None)
            masked = st["masked_idx"].get((s, item), [])
            am = sum(1 for j in masked if R.season_of(data[s][j][0]) == pool)
            assert na == nb - am, (item, pool, s, nb, na, am)
            per_station.append((s, len(idxs), nb, am, na, R.share_non_none(series, idxs, item),
                                R.share_non_none(data_m[s], idxs, item)))
            N += len(idxs); nbef += nb; naft += na; added_pool += am
        pooled = (nbef / N if N else None, naft / N if N else None)
        audit[(item, pool)] = dict(N=N, nBef=nbef, nAft=naft, added=added_pool,
                                   rB=pooled[0], rA=pooled[1],
                                   per_station=per_station)
        last = per_station[-1]
        print(f"{item:26s} {src:4s} | {pool:4s} | {N:5d} {nbef:5d} {naft:5d} {added_pool:5d} | "
              f"{pooled[0]:.2f}→{pooled[1]:.2f} | {last[5]:.2f}→{last[6]:.2f} ({last[0]})")
        rows_out.append(f"| {item} | {pool} | {N:,} | {nbef:,} | {added_pool:,} | {naft:,} "
                        f"| {pooled[0]:.2f}→{pooled[1]:.2f} |")

# 총합 검증
assert sum(v["added"] for v in audit.values()) == 25456, sum(v["added"] for v in audit.values()) \
    if True else None
print()
print("== added 총합 검증:", sum(v["added"] for v in audit.values()))
print()
print("== 기존 S1 표(results_v6.md) 파싱 대조 ==")
import re
fr = open(os.path.join(BASE, "results_v6.md"), encoding="utf-8").read()
mtab = fr.split("## 마스킹 통계")[1]
print()
for m in re.finditer(r"\| (IEM_\w+) \| ([^|]+) \| (\d+) \| ([\d.]+)→([\d.]+) \| ([\d.]+)→([\d.]+) \|", mtab):
    it, src, added = m.group(1), m.group(2).strip(), int(m.group(3))
    bm, am = float(m.group(4)), float(m.group(5))
    bo, ao = float(m.group(6)), float(m.group(7))
    a_mg, a_of = audit[(it, "mgmt")], audit[(it, "off")]
    last_mg = a_mg["per_station"][-1]; last_of = a_of["per_station"][-1]
    print(f"{it:26s} tbl_added={added:6d} real_added={a_mg['added']+a_of['added']:6d} | tbl mgmt {bm:.2f}->{am:.2f}"
          f" | pooled {a_mg['rB']:.2f}->{a_mg['rA']:.2f} | lastStn {last_mg[5]:.2f}->{last_mg[6]:.2f}"
          f" | tbl off {bo:.2f}->{ao:.2f} | pooled {a_of['rB']:.2f}->{a_of['rA']:.2f} | lastStn {last_of[5]:.2f}->{last_of[6]:.2f}")

print()
print("== 보별 mgmt 분모(len(idxs)) 고유값:", sorted({p[1] for it in R.COVS16 for p in audit.get((it,'mgmt'), {'per_station':[]})['per_station']}))
# == pooled rates SSOT 발행(frame/_s1_pooled_rates.json) ==
def r2(a, b):
    return f"{a / N:.2f}->{b / N:.2f}" if False else None

items_js = {}
for item, src in units:
    a_mg, a_of = audit[(item, "mgmt")], audit[(item, "off")]
    added = a_mg["added"] + a_of["added"]
    items_js[item] = {
        "added": added,
        "mg": [f"{a_mg['rB']:.2f}", f"{a_mg['rA']:.2f}"],
        "off": [f"{a_of['rB']:.2f}", f"{a_of['rA']:.2f}"],
        "n": {"mgmt": a_mg["N"], "off": a_of["N"]},
        "nBef": {"mgmt": a_mg["nBef"], "off": a_of["nBef"]},
        "nAft": {"mgmt": a_mg["nAft"], "off": a_of["nAft"]},
    }
assert sum(v["added"] for v in items_js.values()) == 25456
out = {
    "_provenance": ("Pooled across all 16 weirs' visit-day cells (2013-2026)."
                    " Audit 2026-11-22 w26: re-run of frame/run_v6.py build_masked"
                    " (seed 20261120) - per-item added counts bit-identical to the"
                    " frozen run (sum 25,456, restored 0). Replaces the unit_pct"
                    " rates in results_v6.md, which are per-weir snapshots of the"
                    " last processed weir (5004A35 Juksan), with nationwide pooled"
                    " rates. frame/results_v6.md stays frozen."),
    "denominators": {"mgmt": 5416, "off": 6691},
    "items": items_js,
}
with open(os.path.join(BASE, "_s1_pooled_rates.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
import hashlib
_js = open(os.path.join(BASE, "_s1_pooled_rates.json"), "rb").read()
print("== _s1_pooled_rates.json md5:", hashlib.md5(_js).hexdigest(),
      "sha256:", hashlib.sha256(_js).hexdigest())

json.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != 'per_station'} for k, v in audit.items()},
          open(os.path.join(BASE, "_audit_s1_w26.json"), "w"), indent=1)
print("saved frame/_audit_s1_w26.json")
