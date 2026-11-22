# run_v6_16.py — v6 16국 단독 백테스트: run_v4_lag.py 프레임의 16국(보국) 복제
# 데이터: data_weir/algae_pre_measure_bo16_2013_2026.csv (12,107행, 2013.01.02~2026.11.10)
# 변경점 vs v4: ① 로더 load_16(CSV) ② UPSTREAM→16보 체인(NIER 보 개방현황·관리수위 단조 검증)
# ③ wx 도메인 제외(16국 앵커 주간 기상 집계 미확정 — v6 정규 런에서 투입) ④ 나머지 동일
# 기준 동일: 학습≤2023 / 선택·보고=val 2024 / 최종=test 2025–26, 누출 방지 유지
import sys, os, math, random, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import lg
from features import build_xy, train_fill, domain_features, lag_names
from model import ridge_fit, standardize_fit, standardize_apply

BASE = os.path.dirname(os.path.abspath(__file__))
DATA16 = os.path.join(os.path.dirname(BASE), "data_weir", "algae_pre_measure_bo16_2013_2026.csv")
OUT = os.path.join(BASE, "results_v6_16.md")

VAL_Y, TEST_Y = (2024,), (2025, 2026)
HORIZONS = [3, 7, 14]
DOMS = ["hist", "water", "bio", "taste", "up", "lag"]  # v4 − wx
CORE = ["lg_cell_t0", "lg_cell_t1", "mon_sin", "mon_cos"]
EPS, MAX_ADD, K_PERM, SEED = 0.005, 14, 10, 7
N_LAG = len(lag_names())

# 16보 코드→축약명 (NIER 보 개방현황 순서: 한3/낙8/금3/영2)
ST16 = [
    ("1007A20", "강천"), ("1007A27", "여주"), ("1007A60", "이포"),
    ("2007A25", "상주"), ("2009A05", "낙단"), ("2009A30", "구미"), ("2011A25", "칠곡"),
    ("2011A55", "강정고령"), ("2014A25", "달성"), ("2014A70", "합천창녕"), ("2020A32", "창녕함안"),
    ("3012A07", "세종"), ("3012A32", "공주"), ("3012A42", "백제"),
    ("5004A10", "승촌"), ("5004A35", "죽산"),
]
STATIONS16 = [c for c, _ in ST16]
SDN16 = dict(ST16)

# 16보 상류 체인 (하류 방향: 한 강천→여주→이포 / 낙 상주→낙단→구미→칠곡→강정고령→달성→합천창녕→창녕함안
# / 금 세종→공주→백제 / 영 승촌→죽산 — 낙동강 관리수위 47.0→40.0→32.5→25.5→19.5→14.0→10.5→5.0 단조 증거)
UP16 = {"1007A27": "1007A20", "1007A60": "1007A27",
        "2009A05": "2007A25", "2009A30": "2009A05", "2011A25": "2009A30", "2011A55": "2011A25",
        "2014A25": "2011A55", "2014A70": "2014A25", "2020A32": "2014A70",
        "3012A32": "3012A07", "3012A42": "3012A32",
        "5004A35": "5004A10"}

# 로더가 전달할 공변량(도메인 요구 집합 + MICROSTLR 가용성 확인용)
COVS16 = ["IEM_WTRTP", "IEM_PH", "IEM_DOC", "IEM_TUR", "IEM_CHLA",
          "IEM_BGALAGE_MICROSTS", "IEM_BGALAGE_ANBA", "IEM_BGALAGE_OSRTRIA", "IEM_BGALAGE_APZO",
          "IEM_GEOSM", "IEM_MIB2", "IEM_MICROSTLR"]
CELL = "IEM_BGALAGE_CELL_CO"


def load_16():
    """bo16 CSV → {sid: [(date, cell, covs)]} 날짜 오름차순 — core.load_master 동일 형태.
    비수치 토큰('분석중' 2건 실측)·공백은 core.fnum으로 None 처리."""
    with open(DATA16, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {s: [] for s in STATIONS16}
    n_rows = 0
    for r in rows:
        sid = (r.get("SWMN_CODE") or "").strip()
        if sid not in out:
            continue
        d = core._d(r["CHCK_DE"].strip())
        cell = core.fnum(r.get(CELL))
        covs = {k: core.fnum(r.get(k)) for k in COVS16}
        out[sid].append((d, cell, covs))
        n_rows += 1
    for s in out:
        out[s].sort(key=lambda t: t[0])
    return out, n_rows


def rmse(p, t):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)) / len(p))


def cols(rows, js):
    return [[r[j] for j in js] for r in rows]


def pooled(H, data):
    fills = {s: train_fill(data, s, H, DOMS, VAL_Y, TEST_Y) for s in STATIONS16}
    names = domain_features(DOMS)
    X, y, m = [], [], []
    for s in STATIONS16:
        _, X2, y2, m2 = build_xy(s, data[s], data, H, DOMS, VAL_Y, TEST_Y, fill=fills[s])
        X += X2; y += y2; m += m2
    return names, X, y, m


def pred_fn(Xtr, ytr, js, names, X):
    mu, sd = standardize_fit(cols(Xtr, js))
    beta = ridge_fit(standardize_apply(cols(Xtr, js), mu, sd), ytr, 1e-2)
    def pr(rows):
        return [sum(b * v for b, v in zip(beta, list(r) + [1.0]))
                for r in standardize_apply(rows, mu, sd)]
    return beta, pr, (mu, sd)


def domain_of(n):
    if n in CORE: return "hist"
    if n == "lg_cell_t2" or n.endswith("_lag1") or n.endswith("_lag2"): return "lag"
    if n.startswith("wx_"): return "wx"
    if n.startswith("up_"): return "up"
    if n in ("lg_IEM_GEOSM", "lg_IEM_MIB2"): return "taste"
    if "GALAGE" in n: return "bio"
    return "water"


def main(H, body, data):
    import features
    features.UPSTREAM = UP16  # 16보 체인 주입(빌드 시점 조회)
    names, X, y, m = pooled(H, data)
    yg = [lg(v) for v in y]
    ix = {p: [i for i, e in enumerate(m) if e["pool"] == p] for p in ("train", "val", "test")}
    Xtr, ytr = [X[i] for i in ix["train"]], [yg[i] for i in ix["train"]]
    Xva, yva = [X[i] for i in ix["val"]], [yg[i] for i in ix["val"]]
    Xte, yte = [X[i] for i in ix["test"]], [yg[i] for i in ix["test"]]
    if not (Xtr and Xva and Xte):
        raise RuntimeError("empty pool H=%d tr/va/te=%d/%d/%d" % (H, len(Xtr), len(Xva), len(Xte)))

    # --- 발견형 특징 선택(val RMSE 기준, hist 코어 고정) ---
    sel, cand, steps, cur = list(CORE), [n for n in names if n not in CORE], [], None
    def vrm(c):
        js = [names.index(n) for n in c]
        _, pr, _ = pred_fn(Xtr, ytr, js, names, X)
        return rmse(pr(cols(Xva, js)), yva)
    cur = vrm(sel)
    while cand and len(steps) < MAX_ADD:
        best = min((vrm(sel + [c]), c) for c in cand)
        if best[0] >= cur - EPS: break
        sel.append(best[1]); cand.remove(best[1])
        steps.append((best[1], domain_of(best[1]), best[0], best[0] - cur))
        cur = best[0]

    js = [names.index(n) for n in sel]
    beta, pr, (mu, sd) = pred_fn(Xtr, ytr, js, names, X)
    pv, pt = pr(cols(Xva, js)), pr(cols(Xte, js))
    base_va, base_te = rmse([r[0] for r in cols(Xva, [0])], yva), rmse([r[0] for r in cols(Xte, [0])], yte)

    per = {}
    for i in ix["test"]:
        s = m[i]["sid"]; per.setdefault(s, []).append(i)
    per_s = {s: rmse([pt[ix["test"].index(i)] for i in v], [yg[i] for i in v])
             for s, v in sorted(per.items())}

    dl = {}
    for f in list(sel):
        j2 = [names.index(n) for n in sel if n != f]
        _, pr2, _ = pred_fn(Xtr, ytr, j2, names, X)
        dl[f] = rmse(pr(cols(Xva, js)), yva) - rmse(pr2(cols(Xva, j2)), yva)

    random.seed(SEED)
    Z = standardize_apply(cols(Xte, js), mu, sd)
    b_te = rmse(pt, yte)
    perm = {}
    for k, f in enumerate(sel):
        d = []; col = [r[k] for r in Z]
        for _ in range(K_PERM):
            c2 = col[:]; random.shuffle(c2)
            zp = [list(r) for r in Z]
            for i in range(len(zp)): zp[i][k] = c2[i]
            pv2 = [sum(b * v for b, v in zip(beta, list(r) + [1.0])) for r in zp]
            d.append(rmse(pv2, yte) - b_te)
        perm[f] = sum(d) / len(d)

    # --- 보고 ---
    lag_sel = [n for n in sel if domain_of(n) == "lag"]
    body.append(f"\n## H={H} (n train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)})\n")
    body.append("- 선택 순서: " + "; ".join(f"{c}({s})" for c, s, _, _ in steps)
                + (f" — val RMSE {steps[0][2]:.3f}" if steps else ""))
    body.append("- wx: 제외(16국 앵커 주간 기상 집계 미확정 — v6 정규 런에서 투입 예정)")
    body.append(f"- lag 선택 {len(lag_sel)}/{N_LAG}: " + (", ".join(lag_sel) if lag_sel else "없음"))
    body.append(f"- 최종 sel RMSE_log: val {rmse(pv, yva):.3f} / test {b_te:.3f} "
                f"(persistence 동일표본: val {base_va:.3f} / test {base_te:.3f})")
    body.append("- per-station test RMSE_log: "
                + ", ".join(f"{SDN16[s]} {v:.3f}" for s, v in per_s.items()))
    body.append("| 특징 | 도메인 | |β_std| | Δval(drop-one) | Δtest(perm) |")
    body.append("|---|---|---|---|---|")
    for n in sorted(sel, key=lambda n: -perm[n]):
        k = sel.index(n)
        body.append(f"| {n} | {domain_of(n)} | {abs(beta[k]):.3f} | {dl[n]:+.3f} | {perm[n]:+.3f} |")
    dom_sum = {}
    for n in sel:
        dom_sum[domain_of(n)] = dom_sum.get(domain_of(n), 0.0) + perm[n]
    body.append("- 도메인 합계 Δtest(perm): "
                + ", ".join(f"{d} {v:+.3f}" for d, v in sorted(dom_sum.items(), key=lambda kv: -kv[1])))


def smoke():
    """스모크 게이트(exit 코드): H=3·축소(MAX_ADD=3, K_PERM=2)로 전 파이프라인 1회 통과."""
    global MAX_ADD, K_PERM
    MAX_ADD, K_PERM = 3, 2
    data, n_rows = load_16()
    assert n_rows >= 11000, f"rows={n_rows}"
    assert len(data) == 16, f"stations={len(data)}"
    assert min(len(v) for v in data.values()) >= 500, min(len(v) for v in data.values())
    body = ["# v6_16 smoke"]
    main(3, body, data)
    txt = "\n".join(body)
    assert "최종 sel RMSE_log" in txt and "per-station" in txt
    print("SMOKE_OK 16st rows=%d" % n_rows)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
        sys.exit(0)
    data, n_rows = load_16()
    dmin = min(t[0] for v in data.values() for t in v).isoformat()
    dmax = max(t[0] for v in data.values() for t in v).isoformat()
    n_eff = {s: sum(1 for t in data[s] if t[1] is not None) for s in STATIONS16}
    cov_rows = []
    for H in [3]:  # 지점×연도 격자(★법칙 검증 입력) — H=3 pair 기준
        import features
        features.UPSTREAM = UP16
        _, _, _, mp = pooled(H, data)
        cnt = {}
        for e in mp:
            y = e["target"][:4]; p = e["pool"]
            cnt.setdefault(e["sid"], {}).setdefault(y, {"train": 0, "val": 0, "test": 0})
            cnt[e["sid"]][y][p] += 1
        pools = {"train": 0, "val": 0, "test": 0}
        for e in mp:
            pools[e["pool"]] += 1
        cov_rows.append((H, pools, cnt))
    body = ["# v6 16국 단독 백테스트 — v4_lag 프레임 복제(16보 확장, wx 제외)",
            "",
            f"- 데이터: `{os.path.relpath(DATA16, os.path.dirname(BASE))}` ({n_rows}행, {dmin}~{dmax})",
            f"- 지점: 16보(SWMN_CODE) 세포수 유효 {sum(n_eff.values())}/{sum(len(v) for v in data.values())}행 — "
            + ", ".join(f"{SDN16[s]} {n_eff[s]}" for s in STATIONS16),
            "- 유효 규칙: 비수치('분석중' 실측 2건)·공백 = 결측; log10(y+1) 회귀(세포수 0 포함)",
            "- 상류 체인(하류 방향): 한 강천→여주→이포 / 낙 상주→낙단→구미→칠곡→강정고령→달성→합천창녕→창녕함안"
            " / 금 세종→공주→백제 / 영 승촌→죽산 — NIER 보 개방현황 + 관리수위 단조(47.0→5.0) 증거",
            "- 도메인: hist(코어 4)+water(5)+bio(4)+taste(2)+up(2)+lag(" + str(N_LAG) + ") — wx 제외"
            "(16국 앵커 주간 기상 집계 미확정, v6 정규 런에서 투입)",
            f"- 규약: 학습≤2023 / 선택·보고=val 2024 / 최종=test 2025–26; ridge λ=1e-2, "
            f"선택 EPS={EPS}·MAX_ADD={MAX_ADD}, perm K={K_PERM}·seed={SEED} — v4_lag 동일",
            "- 지점×연도 pair 수(test, H=3): 지점별(2025/2026) — " + ", ".join(
                f"{SDN16[s]} {cov_rows[0][2].get(s, {}).get('2025', {'test': 0})['test']}"
                f"/{cov_rows[0][2].get(s, {}).get('2026', {'test': 0})['test']}"
                for s in STATIONS16),
            ""]
    for H in HORIZONS:
        main(H, body, data)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    print("written:", OUT)
