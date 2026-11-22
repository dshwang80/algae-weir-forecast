# probe scratch copy — lgbm_probe.py의 결정론 재실행(P0/P1/P2 전 설정·전 국 dump).
# 원본 probe/는 수정하지 않음. 게이트: abc_H{7,3}.npz sha256 sidecar.
# 판정 규약 동일: native = 전국 train(<=2023) 학습 → 각국 test, LOSO = 각국 drop 후 학습.
import os, sys, json, hashlib
import numpy as np
import lightgbm as lgb

BASE3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(BASE3, "probe")

SETTINGS = {
    "P0": dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.8),
    "P1": dict(num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8),
    "P2": dict(num_leaves=63, min_data_in_leaf=10, feature_fraction=0.8),
}
COMMON = dict(objective="regression", metric="l2", learning_rate=0.05, seed=7, n_jobs=1, verbose=-1,
              bagging_fraction=0.8, bagging_freq=1)
NEST = {"P0": 400, "P1": 300, "P2": 600}

def load_gate(H):
    npz = os.path.join(PROBE, "abc_H%d.npz" % H)
    want = open(npz + ".sha256").read().split()[0].strip()
    got = hashlib.sha256(open(npz, "rb").read()).hexdigest()
    assert got == want, ("sha256 gate FAIL", H)
    return np.load(npz, allow_pickle=False)

def fit_pred(params, n_est, X, ylg, tr, te_rows):
    m = lgb.LGBMRegressor(**params, n_estimators=n_est)
    m.fit(X[tr], ylg[tr])
    return m.predict(X[te_rows])

def rmse(pr, yl, ids):
    return float(np.sqrt(np.mean((pr - yl[ids]) ** 2)))

def run(params, n_est, d, tag):
    X = d["X" + tag.lower()]; ylg = d["ylg"]; sid = d["sid"].astype(str); pool = d["pool"].astype(str)
    te_ids = np.where(pool == "test")[0]; tr_ids = np.where(pool == "train")[0]
    sids = sorted(set(sid[te_ids]))
    pr = fit_pred(params, n_est, X, ylg, tr_ids, te_ids)
    nat = {s: rmse(pr[sid[te_ids] == s], ylg, te_ids[sid[te_ids] == s]) for s in sids}
    nat_m = float(np.mean(list(nat.values())))
    loso = {}
    for s in sids:
        tr_i = np.array([i for i in tr_ids if sid[i] != s])
        te_i = te_ids[sid[te_ids] == s]
        p = fit_pred(params, n_est, X, ylg, tr_i, te_i)
        loso[s] = rmse(p, ylg, te_i)
    loso_m = float(np.mean(list(loso.values())))
    return nat, nat_m, loso, loso_m, sids

out = {}
cross = {"H7": {"P0": (0.898, 0.916), "P1": (0.872, 0.890)},
         "H3": {"P0": (0.667, 0.695)}}
for H in (7, 3):
    d = load_gate(H)
    print("== H%d (n_rows=%d) sha256 gate OK ==" % (H, d["ylg"].shape[0]))
    row = {}
    for tag in ("C", "A", "B"):
        row[tag] = {}
        for sk, sp in SETTINGS.items():
            nat, nat_m, loso, loso_m, sids = run(dict(COMMON, **sp), NEST[sk], d, tag)
            row[tag][sk] = (nat, nat_m, loso, loso_m)
            print("  %s %s: native %.3f | LOSO %.3f" % (tag, sk, nat_m, loso_m))
            exp = cross.get("H%d" % H, {}).get(sk)
            if exp and tag == "C":
                print("      cross-check vs published (C행): pub native %.3f/LOSO %.3f → |Δ| native %.4f LOSO %.4f"
                      % (exp[0], exp[1], abs(nat_m - exp[0]), abs(loso_m - exp[1])))
    out[H] = {tag: {sk: [v[0], v[1], v[2], v[3]] for sk, v in t.items()} for tag, t in row.items()}
    if H == 7:
        for sk in ("P0", "P1", "P2"):
            lc = row["C"][sk][2]; lb = row["B"][sk][2]
            dz = sorted(((s, lb[s] - lc[s]) for s in lc), key=lambda t: -abs(t[1]))
            print("  per-station LOSO B−C %s (top):" % sk)
            for s, v in dz[:4]:
                print("    %s %+.3f (%.3f→%.3f)" % (s, v, lc[s], lb[s]))
json.dump(out, open(os.path.join(PROBE, "_lgbm_v2_full.json"), "w"))
print("dumped: probe/_lgbm_v2_full.json (scratch tree)")
