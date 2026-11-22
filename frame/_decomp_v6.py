"""② 결측 분해 실측 — rtwq 자동측정망 스트림 (v6 마스킹 입력 SSOT 산출). v1 2026-11-20.

입력: ../data_weir/rtwq_auto_daily.csv (444,777행, 71개소, 45 M항목, 2007-01-01~2026-09-30)
출력: results_v6_decomp.md (SSOT) + results_v6_decomp_raw.json (3 마스킹 러너 입력 계약)
정의:
  결측 = 셀 blank(float 파싱 실패 토큰 0건 실측·행 단위 토큰 5,491,561 → 고유 일 셀 5,465,759)
        + 행 미존재 일(격자 구멍, 사이트 창 내 주입).
  중복 (site,date) 2,000행 = 동일(flags) 재수집 실측 → 고유 일로 collapse 후 집계.
  사이트 창 = [첫 행 일, 마지막 행 일] 전체 달력 격자.
  런(run) = 항목(또는 패밀리) 활성 구간[첫 측정일, 마지막 측정일] 내 연속 결측 일수.
  시즌 = 관리기간 6-10월 / 그 밖(11-5월). 런 배정 = 시작일 기준.
  패밀리(러너 cov OR-채움) = 수온[M02,M38,M69], pH[M03,M39,M70], 유기물[M06,M81],
    총인[M28], 탁도[M79,M80,M73], Chla[M29].
"""
import csv
import json
from collections import defaultdict
from datetime import date, timedelta

CSV_PATH = "../data_weir/rtwq_auto_daily.csv"
OUT_MD = "results_v6_decomp.md"
OUT_JSON = "results_v6_decomp_raw.json"

FAMS = [
    ("WTRTP(수온)", ["M02", "M38", "M69"]),
    ("PH", ["M03", "M39", "M70"]),
    ("ORG(유기물)", ["M06", "M81"]),
    ("TRP(총인)", ["M28"]),
    ("TUR(탁도)", ["M79", "M80", "M73"]),
    ("CHLA", ["M29"]),
]
FAM_OF_ITEM = {}
for fam, cols in FAMS:
    for c in cols:
        FAM_OF_ITEM[c] = fam

SEASON_MGMT = "mgmt"   # 6-10월
SEASON_OFF = "off"     # 11-5월
WINDOWS = [
    ("전기간", None, None),
    ("5개년(2022-26)", 2022, 2026),
    ("3개년(2024-26)", 2024, 2026),
]
BUCKETS = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 100), (101, 10**6)]


def bucket_of(n):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n <= hi:
            return i
    raise AssertionError(n)


def season_of(m):
    return SEASON_MGMT if 6 <= m <= 10 else SEASON_OFF

def load():
    items = None
    sites = defaultdict(list)  # sid -> [(date_ord, flags_str)]
    n_rows = 0
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx_date = hdr.index("MSR_DATE")
        idx_sid = hdr.index("SITE_ID")
        item_cols = [i for i, h in enumerate(hdr) if h.startswith("M") and h[1:].isdigit()]
        items = [hdr[i] for i in item_cols]
        idc = {c: k for k, c in enumerate(items)}
        seen = set()   # (sid, date) 중복 collapse(다중행 동일 flags 실측 → 어느 행이든 동일)
        for row in r:
            n_rows += 1
            sid = row[idx_sid]
            d = row[idx_date].strip().replace(".", "-")
            yy, mm, dd = (int(x) for x in d.split("-"))
            o = date(yy, mm, dd).toordinal()
            key = (sid, o)
            if key in seen:
                continue
            seen.add(key)
            flags = "".join("1" if row[i].strip() else "0" for i in item_cols)
            sites[sid].append((o, flags))
    for sid in sites:
        sites[sid].sort(key=lambda t: t[0])
    return items, idc, sites, n_rows, len(seen)


# --- 집계 컨테이너 ---
STAT = {"sites_n": 0, "rows": 0, "grid_holes": 0}
DAYS_ALL = defaultdict(int)          # (year, season) -> 창 내 일수 누적
FILLI = defaultdict(int)             # (item, year, season) -> 채움 일수
FILLF = defaultdict(int)             # (fam, y, s) -> OR 채움 일수
RUN_FAM = {}                         # fam -> {"all":[len...], "mgmt":[...], "off":[...]}
RUN_POOL_ALL = {"all": [], "mgmt": [], "off": []}    # 전체 패밀리 런 합산 풀(season 구분)
YEARS_SEEN = set()


def record_run(fam, start_ord, end_ord):
    n = end_ord - start_ord + 1
    m = date.fromordinal(start_ord).month
    pool = SEASON_MGMT if 6 <= m <= 10 else SEASON_OFF
    RUN_FAM[fam]["all"].append(n)
    RUN_FAM[fam][pool].append(n)
    RUN_POOL_ALL["all"].append(n)
    RUN_POOL_ALL[pool].append(n)


def site_process(sid, rows, items, idc):
    d0, d1 = rows[0][0], rows[-1][0]
    window_days = d1 - d0 + 1
    row_dates = set(o for o, _ in rows)
    STAT["sites_n"] += 1
    STAT["rows"] += len(rows)
    STAT["grid_holes"] += window_days - len(row_dates)
    ni = len(items)
    presents = [set() for _ in items]
    for o, flags in rows:
        for k, ch in enumerate(flags):
            if ch == "1":
                presents[k].add(o)
    for o in range(d0, d1 + 1):
        dt = date.fromordinal(o)
        y, m = dt.year, dt.month
        s = SEASON_MGMT if 6 <= m <= 10 else SEASON_OFF
        YEARS_SEEN.add(y)
        DAYS_ALL[(y, s)] += 1
        if o in row_dates:
            for k in range(ni):
                if o in presents[k]:
                    FILLI[(items[k], y, s)] += 1
            for fam, cols in FAMS:
                for c in cols:
                    k = idc.get(c)
                    if k is not None and o in presents[k]:
                        FILLF[(fam, y, s)] += 1
                        break
    # 런: 패밀리 활성 구간 내 연속 결측
    for fam, cols in FAMS:
        fam_present = set()
        for c in cols:
            k = idc.get(c)
            if k is not None:
                fam_present |= presents[k]
        if fam not in RUN_FAM:
            RUN_FAM[fam] = {"all": [], SEASON_MGMT: [], SEASON_OFF: []}
        if not fam_present:
            continue
        fa, fb = min(fam_present), max(fam_present)
        run_start = None
        for o in range(fa, fb + 1):
            present = o in fam_present
            if not present:
                if run_start is None:
                    run_start = o
            else:
                if run_start is not None:
                    record_run(fam, run_start, o - 1)
                    run_start = None
        if run_start is not None:
            record_run(fam, run_start, fb)

# --- 3단계: 산출 ---
def pctile(sorted_xs, q):
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    k = max(0, min(n - 1, int((q / 100.0) * n + 0.5) - 1))
    return float(sorted_xs[k])


def rate_pct(fam, y, s):
    d = DAYS_ALL[(y, s)]
    return 0.0 if d == 0 else 100.0 * FILLF[(fam, y, s)] / d


def main():
    items, idc, sites, n_rows, rows_uniq = load()
    assert n_rows == 444777, n_rows
    assert rows_uniq == 442777, rows_uniq   # 중복 (site,date) 2,000행 collapse 실측(동일 flags)
    STAT["rows_file"] = n_rows
    STAT["rows_uniq"] = rows_uniq
    STAT["dups"] = n_rows - rows_uniq
    for sid in sorted(sites):
        site_process(sid, sites[sid], items, idc)

    numeric_cells = sum(FILLI.values())
    flag_tokens = numeric_cells  # 고유 일 셀 기준
    assert numeric_cells == 5465759, numeric_cells   # 고유 (site,item,day) 실측(프로브 2종 교차)
    total_window_days = sum(DAYS_ALL.values())
    assert total_window_days == STAT["rows"] + STAT["grid_holes"]
    for fam, pools in RUN_FAM.items():
        assert len(pools["all"]) == len(pools[SEASON_MGMT]) + len(pools[SEASON_OFF]), fam
    gmin = min(rows[0][0] for rows in sites.values())
    gmax = max(rows[-1][0] for rows in sites.values())
    gd0 = date.fromordinal(gmin)
    gd1 = date.fromordinal(gmax)

    years = sorted(YEARS_SEEN)

    def window_rate_mgmt(basis, wa, wb):
        fill = 0
        days = 0
        for y in years:
            if wa is not None and not (wa <= y <= wb):
                continue
            days += DAYS_ALL[(y, SEASON_MGMT)]
            if basis == "all_items":
                for it in items:
                    fill += FILLI[(it, y, SEASON_MGMT)]
            elif basis == "fams":
                for fam, _ in FAMS:
                    fill += FILLF[(fam, y, SEASON_MGMT)]
            else:
                fill += FILLF[(basis, y, SEASON_MGMT)]
        if days == 0:
            return 0.0
        mult = len(items) if basis == "all_items" else (len(FAMS) if basis == "fams" else 1)
        return 100.0 * fill / (days * mult)

    wf = {}
    for name, wa, wb in WINDOWS:
        wf[name] = {
            "all_items": window_rate_mgmt("all_items", wa, wb),
            "fams": window_rate_mgmt("fams", wa, wb),
        }
        for fam, _ in FAMS:
            wf[name][fam] = window_rate_mgmt(fam, wa, wb)
    all_period_all_season_all_items = 100.0 * sum(
        FILLI[(it, y, s)] for it in items for y in years for s in (SEASON_MGMT, SEASON_OFF)
    ) / (total_window_days * len(items))

    L = []
    L.append("# ② 결측 분해 실측 — 수질 자동측정망(rtwq) 스트림 v1 (2026-11-20)")
    L.append("")
    L.append("- 원천: ../data_weir/rtwq_auto_daily.csv (파일 %d행 → 고유 (site,date) %d행·동일 중복 %d행 collapse × %d M항목, %s~%s, %d개소)" % (
        STAT["rows_file"], STAT["rows_uniq"], STAT["dups"], len(items), gd0.isoformat(), gd1.isoformat(), STAT["sites_n"]))
    L.append("- 결측 규칙: 셀 blank(float 파싱 실패 토큰 0건·행 단위 토큰 5,491,561 → 고유 일 셀 5,465,759) + 사이트 창 내 행 미존재 일(격자 구멍 %d일 주입)" % STAT["grid_holes"])
    L.append("- 런: 패밀리 활성 구간[첫/마지막 측정일] 내 연속 결측 일수(시작일 월로 시즌 배정)")
    L.append("- 시즌: 관리기간 6-10월(mgmt) / 그 밖 11-5월(off)")
    L.append("- 항목명 근거: data_weir/fetch_rtwq_auto.py docstring (M02 수온1, M03 pH1, M06 TOC1, M28 총인(TP), M29 클로로필-a, M38-41 수온/pH/EC/DO(2), M69-73 수온/pH/EC/DO/탁도(3), M79-81 탁도1/탁도2/총유기탄소)")
    L.append("- 패밀리(러너 cov OR-채움): " + "; ".join("%s[%s]" % (fam, ",".join(cols)) for fam, cols in FAMS))
    L.append("")
    L.append("## 1. 데이터 개요")
    L.append("- 고유 행 %d(파일 %d, 동일 중복 −%d) | 구멍(행 미존재) %d | 창 일수 합 %d | 셀 수 %d | 수치 셀 %d(=%.1f%%) | blank 셀 %d" % (
        STAT["rows_uniq"], STAT["rows_file"], STAT["dups"], STAT["grid_holes"], total_window_days,
        total_window_days * len(items), numeric_cells,
        100.0 * numeric_cells / (total_window_days * len(items)),
        total_window_days * len(items) - numeric_cells))
    L.append("")
    L.append("## 2. 패밀리×연도 비결측률(%%) — 관리기간 6-10월 (항목×연도×시즌 전체표는 raw.json fills_pct)")
    L.append("| 패밀리 | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for fam, _ in FAMS:
        row = [("%.1f" % rate_pct(fam, y, SEASON_MGMT) if DAYS_ALL[(y, SEASON_MGMT)] else "-") for y in years]
        L.append("| %s | %s |" % (fam, " | ".join(row)))
    L.append("")
    L.append("### 그 밖(11-5월)")
    L.append("| 패밀리 | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for fam, _ in FAMS:
        row = [("%.1f" % rate_pct(fam, y, SEASON_OFF) if DAYS_ALL[(y, SEASON_OFF)] else "-") for y in years]
        L.append("| %s | %s |" % (fam, " | ".join(row)))
    L.append("")
    L.append("## 3. 항목별 전기간 비결측률 순위(수치셀/창 셀, 양 시즌 합산)")
    L.append("- 주의: 항목별 계측 시작 시점이 다른 경우 활성화 전 구간이 0으로 포함되어 전기간 수치가 낮게 나옴 (연도별 상세 = raw.json)")
    item_rates = sorted(
        ((it, 100.0 * sum(FILLI[(it, y, s)] for y in years for s in (SEASON_MGMT, SEASON_OFF)) / total_window_days) for it in items),
        key=lambda t: -t[1])
    for it, r in item_rates:
        L.append("- %s (%s): %.1f%%" % (it, FAM_OF_ITEM.get(it, "-"), r))
    L.append("")
    L.append("## 4. 창별 비결측률(%%) — 관리기간 6-10월 (Decision 3 XX%% 재료)")
    L.append("| basis | 전기간 | 5개년(2022-26) | 3개년(2024-26) |")
    L.append("|---|---|---|---|")
    L.append("| 전체 45항목 pooled | %.1f%% | %.1f%% | %.1f%% |" % (
        wf["전기간"]["all_items"], wf[WINDOWS[1][0]]["all_items"], wf[WINDOWS[2][0]]["all_items"]))
    L.append("| cov 6패밀리 pooled | %.1f%% | %.1f%% | %.1f%% |" % (
        wf["전기간"]["fams"], wf[WINDOWS[1][0]]["fams"], wf[WINDOWS[2][0]]["fams"]))
    for fam, _ in FAMS:
        L.append("| %s | %.1f%% | %.1f%% | %.1f%% |" % (
            fam, wf["전기간"][fam], wf[WINDOWS[1][0]][fam], wf[WINDOWS[2][0]][fam]))
    L.append("")
    L.append("- 전 기간(양 시즌 합산) 전체 45항목 pooled 비결측률 = %.1f%% (인용치 '약 54%%' 대조용)" % all_period_all_season_all_items)
    L.append("")
    L.append("## 5. 결측 런 분포 — 패밀리 × 풀")
    L.append("| 패밀리:풀 | n | mean | p50 | p90 | p95 | max | 1-3일%% | 4-7일%% | 8-14일%% | 15-30일%% | 31-100일%% | 101+일%% | <=3일 누적%% | <=7일 누적%% | <=14일 누적%% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, _ in FAMS + [("ALL", None)]:
        for pool in ("all", SEASON_MGMT, SEASON_OFF):
            if fam == "ALL":
                lens = RUN_POOL_ALL[pool]
            else:
                lens = RUN_FAM[fam][pool]
            if not lens:
                continue
            xs = sorted(lens)
            n = len(xs)
            shares = [100.0 * sum(1 for x in xs if lo <= x <= hi) / n for lo, hi in BUCKETS]
            c3 = 100.0 * sum(1 for x in xs if x <= 3) / n
            c7 = 100.0 * sum(1 for x in xs if x <= 7) / n
            c14 = 100.0 * sum(1 for x in xs if x <= 14) / n
            L.append("| %s:%s | %d | %.1f | %g | %g | %g | %g | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
                fam, pool, n, sum(xs) / n, pctile(xs, 50), pctile(xs, 90), pctile(xs, 95), xs[-1],
                shares[0], shares[1], shares[2], shares[3], shares[4], shares[5], c3, c7, c14))
    L.append("")
    L.append("## 6. ③ 마스킹 러너 입력 계약")
    L.append("- 파일: results_v6_decomp_raw.json — run_pools: {패밀리: {all/mgmt/off: 정수 list(연속 결측 일수, 정렬)}} + run_pool_all {all/mgmt/off}")
    L.append("- 샘플링: cov 특징별로 자기 패밀리 풀에서 중복허용 등확률 1개 추출 = 결측 블록 길이(일); 패밀리 풀 비면 ALL 풀 대체")
    L.append("- 적용 방식(Idealized 보간/Real-time 유지 등)은 설계 SSOT design_v6.md ③을 따름")
    L.append("")
    L.append("## 7. Decision 3 문장 채움 후보 (관리기간 6-10월)")
    L.append("- 후보 A(전체 45항목 pooled): 3개년 = %.1f%%, 5개년 = %.1f%%" % (
        wf[WINDOWS[2][0]]["all_items"], wf[WINDOWS[1][0]]["all_items"]))
    L.append("- 후보 B(cov 6패밀리 pooled): 3개년 = %.1f%%, 5개년 = %.1f%%" % (
        wf[WINDOWS[2][0]]["fams"], wf[WINDOWS[1][0]]["fams"]))
    L.append("- 인용 대조: 전 기간 통합(양 시즌) 전체항목 = %.1f%% (설계 문장 '약 54%%')" % all_period_all_season_all_items)
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    raw = {
        "meta": {"rows": n_rows, "sites": STAT["sites_n"], "items": items,
                 "families": {fam: cols for fam, cols in FAMS},
                 "grid_holes": STAT["grid_holes"], "window_days": total_window_days,
                 "numeric_cells": numeric_cells, "season": {"mgmt": "6-10", "off": "11-5"},
                 "run_def": "family active-span consecutive missing days", "version": "v1"},
        "fills_pct": {
            "item_year_season": {it: {"%d|%s" % (y, s): round(100.0 * FILLI[(it, y, s)] / DAYS_ALL[(y, s)], 3)
                                      for y in years for s in (SEASON_MGMT, SEASON_OFF)} for it in items},
            "fam_year_season": {fam: {"%d|%s" % (y, s): round(100.0 * FILLF[(fam, y, s)] / DAYS_ALL[(y, s)], 3)
                                      for y in years for s in (SEASON_MGMT, SEASON_OFF)} for fam, _ in FAMS},
        },
        "window_fills_pct": wf,
        "run_pools": {fam: {p: sorted(RUN_FAM[fam][p]) for p in ("all", SEASON_MGMT, SEASON_OFF)} for fam in RUN_FAM},
        "run_pool_all": {p: sorted(RUN_POOL_ALL[p]) for p in ("all", SEASON_MGMT, SEASON_OFF)},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, sort_keys=True)
    print("GATE_EXIT_0 sites=%d rows=%d holes=%d cells=%d runs_all=%d md_lines=%d" % (
        STAT["sites_n"], n_rows, STAT["grid_holes"], numeric_cells, len(RUN_POOL_ALL["all"]), len(L)))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
