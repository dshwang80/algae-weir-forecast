# algae-weir-forecast

Deterministic backtest, masking, and transfer-analysis pipeline for the
multi-weir cyanobacteria forecasting study by Kim et al. ("Evaluating
short- to medium-term harmful cyanobacteria forecasts across 16 river weirs
under observed missingness in water-quality data"). A ridge regression forecaster is
compared under a full-information arm, a mask-removal arm that emulates the
automated network's short sampling runs, and a leave-one-station-out
(LOSO) transfer protocol.

## Reproduction tiers

The repository ships the processed inputs needed for exact recomputation
(Tier A). Raw open-API refetch (Tier B) is documented but not automated
here; no fetch scripts or API keys are included in this repository.

- **Tier A — exact recomputation from shipped inputs.** All commands below
  run only on files in this repository and reproduce the manuscript tables
  and prediction dumps deterministically (fixed seeds and deterministic
  feature construction; the LightGBM comparator additionally requires the
  two preprocessed `.npz` inputs, which are shipped).
- **Tier B — raw-data rebuild.** Re-fetching the underlying open datasets
  and re-deriving the processed tables. The sources are Korean national
  open-data APIs (algal pre-measurement service, automated water-quality
  monitoring network, ASOS weather, weir hydraulics); the processed tables
  shipped above were derived from those public sources. No fetch scripts or
  the raw-build script for the mask parameterisation (`_decomp_v6`) are
  included; the column schema of the shipped CSV documents the target
  format, and the shipped `results_v6_decomp_raw.json` / `_s1_pooled_rates.json`
  are the frozen parameterisations the manuscript used.

## Repository layout

```
frame/
  core.py                 station table, ledger logging (lg), helpers
  features.py             build_xy, train_fill, domain_features, lag names
  model.py                ridge fit + standardization
  weather.py              weather feature-name registry (WXD)
  run_v6.py               pooled Full / Real-time / Mask-removal arms -> results_v6.md
  run_v6_16.py            per-weir native engine                -> results_v6_16.md
  run_v6_16_loso.py       leave-one-station-out transfer       -> results_v6_loso.md
  _audit_s1_w26.py        pooled-availability re-audit -> _s1_pooled_rates.json-style checks
  _chk_split_v6.py        chronology-split integrity check
  results_v6*.md          frozen result tables (manuscript reference values)
  results_v6_decomp_raw.json   run-pool mask parameterisation (Tier A input)
  _s1_pooled_rates.json   pooled availability rates (Table S1 reference)
  pred_dump_v6_16_H7.csv  per-record forecast dump, H=7 test window (Table 2/Fig. 3 inputs)
  _boot_arms_r28.py       scenario-difference cluster bootstrap (B=5,000, seed 20261122)
                          -> results_v6_boot.md / _boot_arms_r28.json (Section 3.2 CIs)
data_weir/
  algae_pre_measure_bo16_2013_2026.csv   processed 16-weir observation table (12,107 rows)
probe/
  lgbm_probe_v2.py        LightGBM comparator (Hydrology-5 arm, H = 3/7/14)
  abc_H3.npz / abc_H7.npz  preprocessed comparator inputs (+ .sha256 sidecars)
  _lgbm_v2_full.json      comparator result dump (manuscript S4 / abstract anchor)
  sejong_ablate.py        Seongju (G1/G2/G3)-focused ablation script
  results_lgbm.md         frozen LightGBM diagnostic tables (Table 4 source)
  results_abc.md          frozen A/B/C arm summary (Table 4 / Fig. 5 source)
```

## Environment

Python 3.11. The ridge pipeline (everything under `frame/`) uses the
standard library only. The LightGBM comparator additionally requires:

```
pip install -r requirements.txt   # numpy, lightgbm
```

## Quickstart (Tier A)

Run from the repository root:

```bash
python3 frame/run_v6_16.py          # per-weir native backtest -> frame/results_v6_16.md
python3 frame/run_v6.py             # pooled 3-arm backtest      -> frame/results_v6.md
python3 frame/run_v6_16_loso.py     # LOSO transfer              -> frame/results_v6_loso.md
python3 frame/_audit_s1_w26.py      #availability re-audit of the masking arms
python3 frame/_chk_split_v6.py      # chronology-split check
python3 probe/lgbm_probe_v2.py      # comparator                 -> probe/_lgbm_v2_full.json
```

Each engine rewrites the corresponding `results_*.md` / `_*.json` file in
place. The frozen copies in this repository are the reference values cited
in the manuscript; after a run, compare the regenerated file against the
shipped copy before overwriting it.

## Data notes

- `data_weir/algae_pre_measure_bo16_2013_2026.csv` is the harmonised 16-weir
  observation table (weekly/biweekly pre-measurement service items: water
  temperature, pH, DOC, turbidity, chlorophyll-a, phycocyanin, and
  cyanobacteria cell/genus counts), derived from the Korean national
  algal pre-measurement open service.
- `results_v6_decomp_raw.json` parameterises the mask-removal arm from the
  sampled run-length statistics of the automated water-quality network;
  pooled availability rates are in `frame/_s1_pooled_rates.json`.
- The comparator inputs (`probe/abc_H*.npz`) are preprocessed arrays that
  already incorporate weather-derived and hydraulic features; their raw
  sources (ASOS, weir-gauge open APIs) are not shipped.

## Citation

If you use this code, please cite the manuscript and this repository
(version tag `v1.0.0`). Reproduction scope and additional data
requirements are documented in this README.

## License

MIT (see `LICENSE`).
