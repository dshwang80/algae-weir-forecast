# -*- coding: utf-8 -*-
"""r28 — 검증 시나리오(arm) 간 차이의 불확실성 산출.

run_v6.py의 파이프라인을 그대로 호출해 시나리오별 test 예측을 수집하고,
보(weir) 단위 군집 쌍체 부트스트랩으로
  (a) Full 대 persistence,
  (b) Real-time(블록 마스킹) 대 Full,
  (c) 최악(채수 유래 변수 제거 재학습) 대 Full
의 RMSE_log·F1 차이 95% CI를 계산한다.

기존 결과 파일(results_v6*.md)은 읽기만 하고 수정하지 않는다.
출력: frame/results_v6_boot.md
"""
import os
import sys
import math
import random
import json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.stdout.reconfigure(encoding="utf-8")

import run_v6 as R  # noqa: E402

B_BOOT = 5000
BOOT_SEED = 20261122
HORIZONS = [3, 7, 14]
OUT = os.path.join(BASE, "results_v6_boot.md")

# results_v6.md 동결값 — 재현 게이트
GATE = {3: {"Full": 0.696, "Real-time": 0.703, "Worst@Full": 0.696, "Worst@Real-time": 0.696},
        7: {"Full": 0.983, "Real-time": 0.983, "Worst@Full": 0.983, "Worst@Real-time": 0.983},
        14: {"Full": 1.107, "Real-time": 1.085, "Worst@Full": 1.085, "Worst@Real-time": 1.085}}


def rmse(v):
    return math.sqrt(sum(v) / len(v))


def f1(pred, obs, T):
    tp = sum(1 for p, o in zip(pred, obs) if p >= T and o >= T)
    fp = sum(1 for p, o in zip(pred, obs) if p >= T and o < T)
    fn = sum(1 for p, o in zip(pred, obs) if p < T and o >= T)
    if tp == 0:
        return 0.0
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)


def main():
    data, n_rows = R.load_16()
    data_m, _st = R.build_masked(data)
    scen = [("Full", data, False), ("Real-time", data_m, False),
            ("Worst@Full", data, True), ("Worst@Real-time", data_m, True)]

    res = {}
    for H in HORIZONS:
        res[H] = {}
        import features
        features.UPSTREAM = R.UP16
        names, X, y, m = R.pooled(H, data)
        ixte = [i for i, e in enumerate(m) if e["pool"] == "test"]
        sids = [m[i]["sid"] for i in ixte]
        for label, dat, noiem in scen:
            c = R.main(H, [], dat, label, no_iem=noiem)
            got = round(c["b_te"], 3)
            assert got == GATE[H][label], "gate H=%d %s: %.3f != %.3f" % (H, label, got, GATE[H][label])
            res[H][label] = {"pt": c["pt"], "yte": c["yte"],
                             "pers": [r[0] for r in R.cols(c["Xte"], [0])]}
            print("[ok] H=%d %s test RMSE_log %.3f" % (H, label, got))
        base = res[H]["Full"]
        for label in res[H]:
            assert res[H][label]["yte"] == base["yte"], "target mismatch %s" % label
        assert len(sids) == len(base["yte"]), "sid length"
        res[H]["_sids"] = sids

    lines = ["# r28 — 검증 시나리오 간 차이의 보 단위 군집 부트스트랩 CI",
             "",
             "- 표본: test 2025–2026, 보 16개를 복원추출(군집 부트스트랩) B=%d, seed=%d" % (B_BOOT, BOOT_SEED),
             "- 쌍체: 같은 test 레코드에 대한 두 시나리오의 오차를 짝지어 재표본",
             "- persistence = 앵커(직전 확정 세포수) 그대로 사용하는 기준선(동일 표본)",
             "- 재현 게이트: 12개 시나리오×선행시간 test RMSE_log 전부 results_v6.md와 일치",
             ""]

    for H in HORIZONS:
        sids = res[H]["_sids"]
        byw = {}
        for k, s in enumerate(sids):
            byw.setdefault(s, []).append(k)
        ws = sorted(byw)
        full = res[H]["Full"]
        pairs = [("Full − persistence", full["pt"], full["pers"]),
                 ("Real-time − Full", res[H]["Real-time"]["pt"], full["pt"]),
                 ("Worst@Full − Full", res[H]["Worst@Full"]["pt"], full["pt"]),
                 ("Worst@Real-time − Full", res[H]["Worst@Real-time"]["pt"], full["pt"])]
        obs_tbl = []
        yte = full["yte"]
        for nm, a, b in pairs:
            d_r = rmse([(yte[i] - a[i]) ** 2 for i in range(len(yte))]) - \
                rmse([(yte[i] - b[i]) ** 2 for i in range(len(yte))])
            d_fa = f1(a, yte, R.TH_ATT) - f1(b, yte, R.TH_ATT)
            d_fw = f1(a, yte, R.TH_WARN) - f1(b, yte, R.TH_WARN)
            rng = random.Random(BOOT_SEED)
            br, bfa, bfw = [], [], []
            for _ in range(B_BOOT):
                idx = []
                for _ in range(len(ws)):
                    idx += byw[ws[rng.randrange(len(ws))]]
                ya = [yte[i] for i in idx]
                aa = [a[i] for i in idx]
                bb = [b[i] for i in idx]
                br.append(rmse([(ya[j] - aa[j]) ** 2 for j in range(len(ya))]) -
                          rmse([(ya[j] - bb[j]) ** 2 for j in range(len(ya))]))
                bfa.append(f1(aa, ya, R.TH_ATT) - f1(bb, ya, R.TH_ATT))
                bfw.append(f1(aa, ya, R.TH_WARN) - f1(bb, ya, R.TH_WARN))
            for v in (br, bfa, bfw):
                v.sort()
            lo, hi = int(0.025 * B_BOOT), int(0.975 * B_BOOT)
            obs_tbl.append((nm, d_r, br[lo], br[hi], d_fa, bfa[lo], bfa[hi], d_fw, bfw[lo], bfw[hi]))
            print("H=%2d %-24s ΔRMSE %+0.3f [%+0.3f, %+0.3f]" % (H, nm, d_r, br[lo], br[hi]))
        lines.append("## H = %d (test n = %d)" % (H, len(yte)))
        lines.append("")
        lines.append("| 비교 | ΔRMSE_log | 95% CI | ΔF1 관심 | 95% CI | ΔF1 경계 | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for nm, dr, rl, rh, da, al, ah, dw, wl, wh in obs_tbl:
            lines.append("| %s | %+.3f | [%+.3f, %+.3f] | %+.3f | [%+.3f, %+.3f] | %+.3f | [%+.3f, %+.3f] |"
                         % (nm, dr, rl, rh, da, al, ah, dw, wl, wh))
        lines.append("")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("written:", OUT)
    with open(os.path.join(BASE, "_boot_arms_r28.json"), "w", encoding="utf-8") as f:
        json.dump({str(H): {k: v for k, v in res[H].items() if k != "_sids"} for H in HORIZONS}, f)


if __name__ == "__main__":
    main()
