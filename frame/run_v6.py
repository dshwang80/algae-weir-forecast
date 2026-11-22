# run_v6.py — v6 정규 런: Full / Real-time(마스킹) / 최악(수질 투입불능) 3팔 + Idealized(후속)
# 근거: design_v6.md §1(3팔)·§3(런 분포 마스킹)·§4(NaN 게이트)·§5(F1 타겟)·§8(제2팔 조건부)·§9(SSOT)
# 복제 기반: run_v6_16.py 전문(246행) 판독 후 main/pooled/load_16/domain_of 복제 + 마스킹 팔 추가
# Full 팔 = v6_16과 동일 파이프라인(결측 원본 그대로) — H3 val/test 수치 컨트롤 게이트로 일치 확인
import sys, os, math, random, csv, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import lg
from features import build_xy, train_fill, domain_features, lag_names
from model import ridge_fit, standardize_fit, standardize_apply

BASE = os.path.dirname(os.path.abspath(__file__))
DATA16 = os.path.join(os.path.dirname(BASE), "data_weir", "algae_pre_measure_bo16_2013_2026.csv")
OUT = os.path.join(BASE, "results_v6.md")
DECOMP = os.path.join(BASE, "results_v6_decomp_raw.json")
V616 = os.path.join(BASE, "results_v6_16.md")

VAL_Y, TEST_Y = (2024,), (2025, 2026)
HORIZONS = [3, 7, 14]
DOMS = ["hist", "water", "bio", "taste", "up", "lag"]
CORE = ["lg_cell_t0", "lg_cell_t1", "mon_sin", "mon_cos"]
EPS, MAX_ADD, K_PERM, SEED = 0.005, 14, 10, 7
N_LAG = len(lag_names())
MS_SEED = 20261120            # 마스킹 샘플러 시드(파이프라인 SEED와 별개)
TH_ATT, TH_WARN = 3.0, 4.0    # log10(y+1) ≥3.0=관심(≥999세포) / ≥4.0=경계(≥9,999세포)
TRAIN_CUT = "2023-12-31"

ST16 = [
    ("1007A20", "강천"), ("1007A27", "여주"), ("1007A60", "이포"),
    ("2007A25", "상주"), ("2009A05", "낙단"), ("2009A30", "구미"), ("2011A25", "칠곡"),
    ("2011A55", "강정고령"), ("2014A25", "달성"), ("2014A70", "합천창녕"), ("2020A32", "창녕함안"),
    ("3012A07", "세종"), ("3012A32", "공주"), ("3012A42", "백제"),
    ("5004A10", "승촌"), ("5004A35", "죽산"),
]
STATIONS16 = [c for c, _ in ST16]
SDN16 = dict(ST16)
UP16 = {"1007A27": "1007A20", "1007A60": "1007A27",
        "2009A05": "2007A25", "2009A30": "2009A05", "2011A25": "2009A30", "2011A55": "2011A25",
        "2014A25": "2011A55", "2014A70": "2014A25", "2020A32": "2014A70",
        "3012A32": "3012A07", "3012A42": "3012A32",
        "5004A35": "5004A10"}

COVS16 = ["IEM_WTRTP", "IEM_PH", "IEM_DOC", "IEM_TUR", "IEM_CHLA",
          "IEM_BGALAGE_MICROSTS", "IEM_BGALAGE_ANBA", "IEM_BGALAGE_OSRTRIA", "IEM_BGALAGE_APZO",
          "IEM_GEOSM", "IEM_MIB2", "IEM_MICROSTLR"]
CELL = "IEM_BGALAGE_CELL_CO"
# 마스킹 단위: measured 패밀리 5개(wtrtp/ph/org/tur/chla) — 잔여 7항목은 ALL 풀(item 단위 proxy)
FAM_ITEM = {"IEM_WTRTP": "WTRTP(수온)", "IEM_PH": "PH", "IEM_DOC": "ORG(유기물)",
            "IEM_TUR": "TUR(탁도)", "IEM_CHLA": "CHLA"}
M_UNITS = [(i, FAM_ITEM[i]) if i in FAM_ITEM else (i, "ALL") for i in COVS16]
SEASON_MONTHS = set(range(6, 11))  # 관리기간 6-10월(decomp 규약)


def load_16():
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

# ─── 마스킹(Real-time 팔): decomp 런 분포 샘플링 ─────────────────────────
def season_of(d):
    return "mgmt" if d.month in SEASON_MONTHS else "off"


def _mn(seq):
    v = [x for x in seq if x is not None]
    return sum(v) / len(v) if v else None


def load_decomp():
    with open(DECOMP, encoding="utf-8") as f:
        dj = json.load(f)
    rp, rpa = dj["run_pools"], dj["run_pool_all"]
    fy = dj["fills_pct"]["fam_year_season"]
    iy = dj["fills_pct"]["item_year_season"]
    yr = (2024, 2025, 2026)
    tgt_fam = {}
    for fam in rp:
        for pool in ("mgmt", "off"):
            t = _mn([fy[fam].get("%d|%s" % (y, pool)) for y in yr])
            assert t is not None, "decomp target fill missing: %s|%s" % (fam, pool)
            tgt_fam[(fam, pool)] = t / 100.0  # decomp fills_pct = 퍼센트 → 비율 환산(r_ex 동일 스케일)
    tgt_all = {}
    for pool in ("mgmt", "off"):
        seq = []
        for k, e in iy.items():
            m = _mn([e.get("%d|%s" % (y, pool)) for y in yr])
            if m is not None:
                seq.append(m)
        tgt_all[pool] = _mn(seq) / 100.0  # 동일 퍼센트→비율 환산
    for k, v in tgt_fam.items():
        assert v is not None, "decomp target fill missing: %s" % (k,)
    return rp, rpa, tgt_fam, tgt_all


def share_non_none(series, idxs, item):
    if not idxs:
        return 0.0
    return sum(1 for i in idxs if series[i][2].get(item) is not None) / len(idxs)


def mask_station(series, s, units, rp, rpa, tgt_fam, tgt_all, rng, stats):
    """units: [(item, fam 또는 "ALL")] — 기존 결측 유지, 런 분포 블록 가첨 마스킹."""
    out = [(d, c, dict(cv)) for d, c, cv in series]
    for item, src in units:
        stats["per_item"].setdefault(item, 0)
        for pool in ("mgmt", "off"):
            runs = rpa[pool] if src == "ALL" else rp[src][pool]
            if not runs:
                continue
            idxs = [i for i, (d, c, cv) in enumerate(out) if season_of(d) == pool]
            r_ex = share_non_none(series, idxs, item)
            r_t = tgt_all[pool] if src == "ALL" else tgt_fam[(src, pool)]
            skip = (r_t is None or r_ex <= 0.02 or r_t >= r_ex or (r_t / max(r_ex, 1e-9)) > 0.999)
            if not skip:
                r_add = r_t / r_ex
                i = 0
                while i < len(idxs):
                    m = rng.choice(runs)
                    g = int(round(m * r_add / (1.0 - r_add)))
                    i += g
                    for j in idxs[i:i + m]:
                        d, c, cv = out[j]
                        if cv.get(item) is not None:
                            stats["per_item"][item] += 1
                            stats["masked_idx"].setdefault((s, item), []).append(j)
                        cv[item] = None
            stats["unit_pct"].setdefault(item, {})[pool] = (r_ex, share_non_none(out, idxs, item))
    # ★게이트: 마스킹이 학습기간(≤2023) 항목을 전량 소멸시킨 경우만 복구(train_fill 빈 중앙값 방지)
    # — 스테이션×항목이 원자료부터 완전결측인 경우(v6_16과 동일 상태)는 유지
    restored = 0
    for item in COVS16:
        tr = [i for i, (d, c, cv) in enumerate(out)
              if d.isoformat() <= TRAIN_CUT and cv.get(item) is not None]
        if tr:
            continue
        ms = stats["masked_idx"].get((s, item), [])
        if not ms:
            continue  # 마스킹 이전부터 완전결측(v6_16 동일) — 복구 불필요
        out[ms[-1]][2][item] = series[ms[-1]][2].get(item)
        restored += 1
    stats["restored_total"] += restored
    return out


def build_masked(data):
    """전 지점 마스킹 적용 → (data_masked, stats) — 시드 고정·지점/항목/시즌 순회 순서 고정."""
    rp, rpa, tgt_fam, tgt_all = load_decomp()
    rng = random.Random(MS_SEED)
    units = [(k, FAM_ITEM[k]) for k in COVS16 if k in FAM_ITEM] + \
            [(k, "ALL") for k in COVS16 if k not in FAM_ITEM]
    stats = {"per_item": {}, "unit_pct": {}, "masked_idx": {}, "restored_total": 0}
    data_m = {}
    for s in STATIONS16:
        data_m[s] = mask_station(data[s], s, units, rp, rpa, tgt_fam, tgt_all, rng, stats)
    data_m["_meta"] = {"units": units, "tgt_fam": tgt_fam, "tgt_all": tgt_all}
    return data_m, stats


def f1_at(preds, ys, T):
    tp = fp = fn = 0
    for p, y in zip(preds, ys):
        a, b = p >= T, y >= T
        if a and b:
            tp += 1
        elif a:
            fp += 1
        elif b:
            fn += 1
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if (pr + rc) else 0.0), tp + fn


def main(H, body, data, arm, no_iem=False):
    """run_v6_16.py main() 복제 + F1 라인 보강. no_iem=True: 후보에서 IEM*(수질·조류·취미, lg_IEM*·up_lg_IEM* 포함) 전량 제외(최악 팔 재학습)."""
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
    sel, cand, steps, cur = list(CORE), \
        [n for n in names if n not in CORE and not (no_iem and "IEM" in n)], [], None
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
    body.append(f"\n## [{arm}] H={H} (n train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)})\n")
    body.append("- 선택 순서: " + "; ".join(f"{c}({s})" for c, s, _, _ in steps)
                + (f" — val RMSE {steps[0][2]:.3f}" if steps else ""))
    body.append("- wx: 제외(16국 앵커 주간 기상 집계 미확정 — v6 정규 런에서 투입 예정)")
    body.append(f"- lag 선택 {len(lag_sel)}/{N_LAG}: " + (", ".join(lag_sel) if lag_sel else "없음"))
    body.append(f"- 최종 sel RMSE_log: val {rmse(pv, yva):.3f} / test {b_te:.3f} "
                f"(persistence 동일표본: val {base_va:.3f} / test {base_te:.3f})")
    fa, fw = f1_at(pt, yte, TH_ATT), f1_at(pt, yte, TH_WARN)
    body.append(f"- F1(test): 관심 P {fa[0]:.3f} R {fa[1]:.3f} **F1 {fa[2]:.3f}** (pos {fa[3]}; log10(y+1)≥{TH_ATT})"
                f" / 경계 P {fw[0]:.3f} R {fw[1]:.3f} **F1 {fw[2]:.3f}** (pos {fw[3]}; ≥{TH_WARN})")
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
    return {"b_te": b_te, "val": rmse(pv, yva), "pv": pv, "pt": pt, "yte": yte, "js": js, "sel": sel,
            "beta": beta, "mu": mu, "sd": sd, "Xte": Xte, "fa": fa, "fw": fw}


def dtab(body, cf, cr, cw, cwr):
    """팔 비교 Δ표: Full / Real-time / 최악(재학습·IEM 후보 제거) — Full·RT 각각 데이터로 재선택·재학습."""
    body.append("| 팔 | test RMSE_log | F1@관심 | F1@경계 | R@관심 | R@경계 | ΔF1@관심(대 Full) | ΔF1@경계(대 Full) |")
    body.append("|---|---|---|---|---|---|---|---|")
    body.append(f"| Full | {cf['b_te']:.3f} | {cf['fa'][2]:.3f} | {cf['fw'][2]:.3f} "
                f"| {cf['fa'][1]:.3f} | {cf['fw'][1]:.3f} | — | — |")
    body.append(f"| Real-time(마스킹) | {cr['b_te']:.3f} | {cr['fa'][2]:.3f} | {cr['fw'][2]:.3f} "
                f"| {cr['fa'][1]:.3f} | {cr['fw'][1]:.3f} "
                f"| {cr['fa'][2]-cf['fa'][2]:+.3f} | {cr['fw'][2]-cf['fw'][2]:+.3f} |")
    body.append(f"| 최악@Full(재학습·IEM 후보 제거) | {cw['b_te']:.3f} | {cw['fa'][2]:.3f} | {cw['fw'][2]:.3f} "
                f"| {cw['fa'][1]:.3f} | {cw['fw'][1]:.3f} "
                f"| {cw['fa'][2]-cf['fa'][2]:+.3f} | {cw['fw'][2]-cf['fw'][2]:+.3f} |")
    body.append(f"| 최악@Real-time(재학습) | {cwr['b_te']:.3f} | {cwr['fa'][2]:.3f} | {cwr['fw'][2]:.3f} "
                f"| {cwr['fa'][1]:.3f} | {cwr['fw'][1]:.3f} "
                f"| {cwr['fa'][2]-cr['fa'][2]:+.3f} | {cwr['fw'][2]-cr['fw'][2]:+.3f} |")
    note = "- 최악(재학습) sel: " + ", ".join(cw["sel"])
    note += "; Full sel과 동일" if cw["sel"] == cf["sel"] else \
        "; Full sel과 상이(" + ", ".join(cf["sel"]) + ")"
    if cwr["sel"] != cw["sel"]:
        note += " / RT: " + ", ".join(cwr["sel"])
    body.append(note)


def masking_table(body, st):
    body.append("## 마스킹 통계(전 기간, 항목×시즌 대치율 before→after; 가첨 결측 수)")
    body.append("| 항목(마스킹 단위) | 소스 | added행 | mgmt율 before→after | off율 before→after |")
    body.append("|---|---|---|---|---|")
    for item, src in M_UNITS:
        s = st["unit_pct"].get(item, {})
        def f2(p):
            return "%s→%s" % ("%.2f" % p[0], "%.2f" % p[1]) if p else "—"
        body.append(f"| {item} | {src} | {st['per_item'].get(item, 0)} | "
                    f"{f2(s.get('mgmt'))} | {f2(s.get('off'))} |")
    added = sum(st["per_item"].values())
    body.append(f"- 합계 added(가첨 결측 행수): {added}, train 중앙값 보존 복구: {st['restored_total']}")
    return added


def smoke():
    """스모크 게이트(exit 코드): H=3·축소(MAX_ADD=3, K_PERM=2)로 전 파이프라인 1회 통과."""
    global MAX_ADD, K_PERM
    MAX_ADD, K_PERM = 3, 2
    data, n_rows = load_16()
    assert n_rows >= 11000, f"rows={n_rows}"
    assert len(data) == 16, f"stations={len(data)}"
    data_m, st = build_masked(data)
    assert st["restored_total"] == 0, st["restored_total"]
    added = masking_table(body := [], st)
    assert added > 0, "마스킹 추가 0 — decomp 런 풀/타겟 대치율 확인 필요"
    cf = main(3, body, data, "Full")
    cr = main(3, body, data_m, "Real-time")
    cw = main(3, [], data, "최악", no_iem=True)
    cwr = main(3, [], data_m, "최악", no_iem=True)
    dtab(body, cf, cr, cw, cwr)
    txt = "\n".join(body)
    assert "최종 sel RMSE_log" in txt and "F1(test)" in txt and "마스킹 통계" in txt \
        and "최악(재학습) sel" in txt
    print("SMOKE_OK rows=%d added=%d restored=%d" % (n_rows, added, st["restored_total"]))


def ctrl_gate_full_h3(b_te3, val3, v616_path):
    """Full 팔 컨트롤 게이트: H3 val/test 수치 == results_v6_16.md H3 수치(v6_16과 동일 파이프라인 증명)."""
    t = open(v616_path, encoding="utf-8").read()
    m = re.search(r"## H=3 \(n train.*?최종 sel RMSE_log: val ([\d.]+) / test ([\d.]+)", t, re.S)
    assert m, "results_v6_16.md H3 섹션 파싱 실패"
    assert val3 == m.group(1) and b_te3 == m.group(2), \
        "Full H3 ≠ v6_16 H3: %s/%s vs %s/%s" % (val3, b_te3, m.group(1), m.group(2))


if __name__ == "__main__":
    import re
    if "--smoke" in sys.argv:
        smoke()
        sys.exit(0)
    data, n_rows = load_16()
    assert n_rows >= 11000, f"rows={n_rows}"
    assert len(data) == 16, f"stations={len(data)}"
    data_m, st = build_masked(data)
    md5d = hashlib.md5(open(DECOMP, "rb").read()).hexdigest()[:8]
    body = [
        "# v6 Full/Real-time/최악 마스킹 백테스트 — design_v6.md §1 3팔 (16보 CSV, wx 제외)",
        "",
        f"- 데이터: `algae_pre_measure_bo16_2013_2026.csv` ({n_rows}행, 16보)",
        "- Full 팔: v6_16과 동일 파이프라인·결측 원본 그대로",
        ("  — 컨트롤 게이트: H3 val/test 수치 == `results_v6_16.md` H3(전과정 수일치 확인, "
         f"MAX_ADD={MAX_ADD}·K_PERM={K_PERM}·SEED={SEED} 동일)"),
        (f"- Real-time 팔: 런 분포 마스킹(§3) — 소스 `results_v6_decomp_raw.json`(md5 {md5d}), "
         f"런 풀 rpa(ALL 프록시)·rp(fam), 시즌 블록 길이 m=실측 런 분포 샘플링"
         f"(mgmt=6-10월·off=5/11-12/1-4월), 갭 g=m·r_add/(1−r_add), "
         f"r_add=r_target/r_ex(유효율 목표: decomp 3개년(2024–26) fam×시즌 평균 / 보16 관측), "
         f"시드 {MS_SEED}, 기존 결측 유지·가첨만(겹침 없음), 학습기간(≤2023) 항목별 마지막 1행 복구 게이트"),
        "- 최악 팔: 재학습형 — 훈련 후보에서 IEM*(수질·조류·취미; lg_IEM*·up_lg_IEM* 포함) 전량 제외 후 재선택·재학습(Full/Real-time 각각 데이터), 시험시점 대치 아님(2026-11-20 (b) 결정)",
        f"- F1 임계: log10(y+1) ≥ {TH_ATT}(관심, 세포수≥999) / ≥ {TH_WARN}(경계, 세포수≥9,999)",
        "- Idealized 팔(기상 미래 투입): 미시행 — ASOS 앵커 wx 집계 확정 후 별도 런(design_v6.md §8 조건부)",
        "- 규약: 학습≤2023 / 선택·보고=val 2024 / 최종=test 2025–26; ridge λ=1e-2, EPS={0}·MAX_ADD={1}, perm K={2}·seed={3} — v4_lag 동일".format(EPS, MAX_ADD, K_PERM, SEED),
        "",
    ]
    masking_table(body, st)
    body.append("")
    for H in HORIZONS:
        cf = main(H, body, data, "Full")
        cr = main(H, body, data_m, "Real-time")
        cw = main(H, [], data, "최악", no_iem=True)
        cwr = main(H, [], data_m, "최악", no_iem=True)
        dtab(body, cf, cr, cw, cwr)
        if H == 3:
            ctrl_gate_full_h3("%.3f" % cf["b_te"], "%.3f" % cf["val"], V616)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    print("written:", OUT, "added=", sum(st["per_item"].values()))
