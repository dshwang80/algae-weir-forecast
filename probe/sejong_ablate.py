#! /usr/bin/python3
# probe/sejong_ablate.py — 세종보(2007A70) A−C +0.173 원인 분해:
# cov16(수문 4특징) 단독/소거 LOSO fold(H=7) — run_abc_probe 부품 재사용, 판정 코드 불변.
import sys, os
BASE3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE3, "scripts"))
import run_abc_probe as R    # noqa: E402
import run_v6_16 as B        # noqa: E402

ST, H = "3012A07", 7  # 세종(3012A07) — 금강 체계, test n=86
FULL = [ev_i for ev_i in range(1)]  # placeholder replaced below
NAMES = ["hy_lg_V7", "hy_lg_Q7", "hy_lg_I7", "hy_lg_tau7"]

data, _ = B.load_16()
ev = R.build_ext(H, data)
full = [ev["names"].index("hy_lg_" + k) for k in ("V7", "Q7", "I7", "tau7")]
te = [i for i, e in enumerate(ev["m"]) if e["pool"] == "test"]
tr = [i for i, e in enumerate(ev["m"]) if e["pool"] == "train"]
trL = [i for i in tr if ev["m"][i]["sid"] != ST]
ids = [i for i in te if ev["m"][i]["sid"] == ST]

print("세종(2007A70) LOSO fold, H=7 — RMSE_log (C 기준)")
print("%-11s %s" % ("구성", "rmse"))
combos = [("C(기준)", [])]
combos += [("+%s" % n[6:], [j]) for n, j in zip(NAMES, full)]
combos.append(("A(VQIT)", list(full)))
for k in range(4):
    combos.append(("A−%s" % NAMES[k][6:], [j for j in full if j != full[k]]))
for tag, extra in combos:
    js = list(ev["js"]["C"]) + extra
    pr = R.fit_row(js, ev["X"], ev["ylg"], trL)
    rl, f1 = R.stat_row(pr, js, ev["X"], ev["ylg"], ev["y"], ids)
    print("%-11s %.3f  (n=%d, f1_10k %.3f)" % (tag, rl, len(ids), f1))
