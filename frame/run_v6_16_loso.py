# run_v6_16_loso.py — v6 16보 LOSO 프로브: Table 3 Panel A 동일세대 클론
# 프로토콜(v1 results_v1.md LOSO 이식): 학습(train≤2023) 우주에서 drop국 제외,
#   drop국은 test(2025–26)만 평가. drop국 자체 인풋(own lag lg_cell_t0/t1, 상류
#   up_lg_cell 실측, 국별 fill=train구간 중앙값)은 시점 t 실측 인풋으로 허용(v1 규약 동일).
# 특징집합 = results_v6_16.md H=3·7 확정 sel(lg_cell_t0·lg_cell_t1·mon_sin·mon_cos·up_lg_cell)
#   고정 — LOSO에서 선택 재실행 없음(v1 도메인 고정 규약의 대응물).
# native 검증: 동일 특징·데이터로 all-16 재학습(Panel A 재산출) 후 results_v6_16.md
#   H=7 per-station RMSE_log와 hard assert(|Δ|≤0.001, 반올림 허용) — 불일치 시 중단.
import sys, os, math, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features
from features import build_xy, train_fill, domain_features
from model import ridge_fit, standardize_fit, standardize_apply
from core import lg, eval_all
import run_v6_16 as B

SEL = ["lg_cell_t0", "lg_cell_t1", "mon_sin", "mon_cos", "up_lg_cell"]
SDI = {nm: code for code, nm in B.SDN16.items()}
OUT = os.path.join(B.BASE, "results_v6_loso.md")
TOL = 0.0011


def build(H, data):
    """전 지점 행을 1회 빌드(특징 고정) — fold 차이는 train 인덱스 마스킹뿐."""
    features.UPSTREAM = B.UP16
    fills = {s: train_fill(data, s, H, B.DOMS, B.VAL_Y, B.TEST_Y) for s in B.STATIONS16}
    names = domain_features(B.DOMS)
    js = [names.index(n) for n in SEL]
    X, yr, mr = [], [], []
    for s in B.STATIONS16:
        _, X2, y2, m2 = build_xy(s, data[s], data, H, B.DOMS, B.VAL_Y, B.TEST_Y, fill=fills[s])
        X += X2; yr += y2; mr += m2
    yl = [lg(v) for v in yr]
    te = [i for i, e in enumerate(mr) if e["pool"] == "test"]
    tr = [i for i, e in enumerate(mr) if e["pool"] == "train"]
    return js, X, yl, yr, mr, te, tr


def fit(js, X, yl, tr):
    A = [[X[i][j] for j in js] for i in tr]
    mu, sd = standardize_fit(A)
    beta = ridge_fit(standardize_apply(A, mu, sd), [yl[i] for i in tr], 1e-2)
    return lambda rows: [sum(b * v for b, v in zip(beta, list(r) + [1.0]))
                         for r in standardize_apply(rows, mu, sd)]


def stat(pr, js, X, yl, yr, ids):
    plg = pr([[X[i][j] for j in js] for i in ids])
    rl = math.sqrt(sum((a - b) ** 2 for a, b in zip(plg, [yl[i] for i in ids])) / len(ids))
    praw = [max(10 ** v - 1, 0) for v in plg]
    e = eval_all(praw, [yr[i] for i in ids])
    return rl, e["f1_10k"]


def native_ref(H):
    """results_v6_16.md에서 H 지표 per-station RMSE_log 파싱(검증 기준값)."""
    txt = open(os.path.join(B.BASE, "results_v6_16.md"), encoding="utf-8").read()
    sec = re.search(r"## H=%d .*?(?=\Z|\n## )" % H, txt, re.S).group(0)
    line = [s for s in sec.splitlines() if s.startswith("- per-station")][0]
    out = {}
    for pair in line.split(": ", 1)[1].split(", "):
        nm, v = pair.rsplit(" ", 1)
        out[SDI[nm]] = float(v)
    assert len(out) == 16, f"ref parse H={H}: {len(out)}"
    return out


def run_h(H, data):
    js, X, yl, yr, mr, te, tr = build(H, data)
    ref = native_ref(H)
    prnat = fit(js, X, yl, tr)
    nat = {}
    for s in B.STATIONS16:
        ids = [i for i in te if mr[i]["sid"] == s]
        assert ids, f"no test rows: {s}"
        nat[s] = (*stat(prnat, js, X, yl, yr, ids), len(ids))
        d = abs(nat[s][0] - ref[s])
        assert d <= TOL, f"native assert H={H} {s}: probe {nat[s][0]:.3f} vs doc {ref[s]:.3f}"
    rows, mean_L, mean_N = [], 0.0, 0.0
    for s in B.STATIONS16:
        trL = [i for i in tr if mr[i]["sid"] != s]
        assert len(trL) > 20, f"tiny train fold: {s}"
        prL = fit(js, X, yl, trL)
        ids = [i for i in te if mr[i]["sid"] == s]
        rl, f1 = stat(prL, js, X, yl, yr, ids)
        rows.append((s, len(ids), nat[s][0], rl, rl - nat[s][0], f1, nat[s][1]))
        mean_L += rl; mean_N += nat[s][0]
    n16 = len(B.STATIONS16)
    return rows, mean_L / n16, mean_N / n16


def main():
    data, _n = B.load_16()
    out = ["# v6 16국 LOSO — Table 3 Panel B 재실행 프로브(Panel A 동일세대)",
           "",
           "- 규약(v1 results_v1.md LOSO 이식): 학습(train≤2023)에서 drop국 제외, drop국은",
           "  test(2025–26)만 평가. drop국 자체 인풋(own lag t0/t1, 상류 up_lg_cell 실측,",
           "  국별 fill=train구간 중앙값)은 시점 t 실측 인풋으로 허용. ridge λ=1e-2.",
           "- 특징 고정: lg_cell_t0, lg_cell_t1, mon_sin, mon_cos, up_lg_cell",
           "  (results_v6_16.md H=3·7 sel 재사용 — LOSO에서 선택 재실행 없음).",
           "- native 대조: 동일 파이프라인 all-16 재산출 == results_v6_16.md per-station",
           "  RMSE_log (16/16 |Δ|≤0.001 assert 통과)."]
    for H in (7, 3):
        rows, mL, mN = run_h(H, data)
        out += ["", f"## H={H} (test n 합계={sum(r[1] for r in rows)})",
                "| 지점 | n | native RMSE_log | LOSO RMSE_log | Δ(LOSO−native) | LOSO F1@10k | native F1@10k |",
                "|---|---|---|---|---|---|---|"]
        for s, n, nr, rl, d, f1, nf in rows:
            out.append(f"| {B.SDN16[s]} | {n} | {nr:.3f} | {rl:.3f} | {d:+.3f} | {f1:.2f} | {nf:.2f} |")
        out.append(f"- 지점평균 RMSE_log: native {mN:.3f} → LOSO {mL:.3f} ({mL - mN:+.3f})")
        print(f"[H={H}] mean native {mN:.3f} -> LOSO {mL:.3f} ({mL - mN:+.3f}, {(mL - mN) / mN * 100:+.1f}%)")
        for s, n, nr, rl, d, f1, nf in rows:
            print(f"  {B.SDN16[s]:6} n={n:3} native {nr:.3f} loso {rl:.3f} d={d:+.3f} F1loso {f1:.2f}")
    open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("written:", OUT)


if __name__ == "__main__":
    main()
