import os
import csv
import argparse
import itertools

import numpy as np
from scipy import stats

from metrics import METRIC_HIGHER_BETTER

N_BOOT = 10000
DEFAULT_METRICS = ["PESQ", "STOI", "ESTOI", "SISDR", "SegSNR",
                   "CSIG", "CBAK", "COVL", "PhaseD", "GDD", "IFD", "cSER", "LSD"]


def load_per_utt(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["file"]
            d[fn] = {k: (float(v) if v not in ("", None) else np.nan)
                     for k, v in row.items() if k != "file"}
    return d


def paired_arrays(da, db, metric):
    common = sorted(set(da) & set(db))
    a, b = [], []
    for f in common:
        va, vb = da[f].get(metric, np.nan), db[f].get(metric, np.nan)
        if not (np.isnan(va) or np.isnan(vb)):
            a.append(va); b.append(vb)
    return np.array(a), np.array(b)


def _wilcoxon(a, b):
    from scipy import stats as _st
    for kw in ({"method": "approx"}, {"mode": "approx"}, {}):
        try:
            return _st.wilcoxon(a, b, zero_method="wilcox",
                                alternative="two-sided", **kw)
        except TypeError:
            continue
    raise RuntimeError("scipy.stats.wilcoxon 调用失败，请检查 scipy 版本")


def _shapiro(diff):
    from scipy import stats as _st
    if len(diff) < 3:
        return np.nan
    if len(diff) > 5000:
        rng = np.random.default_rng(0)
        diff = rng.choice(diff, 5000, replace=False)
    try:
        return float(_st.shapiro(diff)[1])
    except Exception:
        return np.nan


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else np.nan


def cliffs_delta(a, b):
    n = len(a)
    if n > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 2000, replace=False)
        a, b = a[idx], b[idx]
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_ci(diff, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_bonferroni(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        prev = max(prev, val)
        adj[i] = min(prev, 1.0)
    return adj, adj < alpha


def compare(da, db, name_a, name_b, metrics):
    out = []
    for mt in metrics:
        a, b = paired_arrays(da, db, mt)
        if len(a) < 8:
            continue
        diff = a - b
        higher_better = METRIC_HIGHER_BETTER.get(mt, True)


        sw_p = _shapiro(diff)

        t_stat, t_p = stats.ttest_rel(a, b)

        if np.allclose(diff, 0):
            w_stat, w_p = np.nan, 1.0
        else:
            w_stat, w_p = _wilcoxon(a, b)

        lo, hi = bootstrap_ci(diff)
        out.append(dict(
            metric=mt, N=len(a),
            mean_a=a.mean(), std_a=a.std(ddof=1),
            mean_b=b.mean(), std_b=b.std(ddof=1),
            delta=diff.mean(), ci_lo=lo, ci_hi=hi,
            shapiro_p=sw_p, t_p=float(t_p), wilcoxon_p=float(w_p),
            dz=cohens_dz(diff), cliff=cliffs_delta(a, b),
            better=("A" if (diff.mean() > 0) == higher_better else "B"),
            higher_better=higher_better,
        ))
    return out


def stars(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="本文方法的逐条 CSV")
    ap.add_argument("--b", type=str, nargs="+", required=True, help="一个或多个对比方法 CSV")
    ap.add_argument("--name_a", type=str, default="Proposed")
    ap.add_argument("--name_b", type=str, nargs="*", default=None)
    ap.add_argument("--metrics", type=str, nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--out", type=str, default="results/stats_report.csv")
    args = ap.parse_args()

    da = load_per_utt(args.a)
    names_b = args.name_b or [os.path.basename(p).replace("per_utt_", "").replace(".csv", "")
                              for p in args.b]

    all_rows = []
    for path_b, nb in zip(args.b, names_b):
        db = load_per_utt(path_b)
        for r in compare(da, db, args.name_a, nb, args.metrics):
            r["vs"] = nb
            all_rows.append(r)


    pv = np.array([r["wilcoxon_p"] for r in all_rows])
    valid = ~np.isnan(pv)
    adj = np.full_like(pv, np.nan)
    if valid.sum() > 0:
        adj_v, _ = holm_bonferroni(pv[valid])
        adj[valid] = adj_v
    for r, p in zip(all_rows, adj):
        r["wilcoxon_p_holm"] = float(p)

    for nb in names_b:
        sub = [r for r in all_rows if r["vs"] == nb]
        if not sub:
            continue
        print("\n" + "=" * 108)
        print("%s  vs  %s   (N=%d 条配对测试语音)" % (args.name_a, nb, sub[0]["N"]))
        print("=" * 108)
        print("%-8s %18s %18s %20s %11s %10s %8s" %
              ("指标", args.name_a[:16], nb[:16], "Δ (95%CI)",
               "Wilcoxon", "Holm校正", "dz"))
        print("-" * 108)
        for r in sub:
            arrow = "↑" if r["higher_better"] else "↓"
            print("%-6s%s %8.4f±%-8.4f %8.4f±%-8.4f %+7.4f [%+.4f,%+.4f] %8.2e%-3s %8.2e%-3s %7.3f" % (
                r["metric"], arrow,
                r["mean_a"], r["std_a"], r["mean_b"], r["std_b"],
                r["delta"], r["ci_lo"], r["ci_hi"],
                r["wilcoxon_p"], stars(r["wilcoxon_p"]),
                r["wilcoxon_p_holm"], stars(r["wilcoxon_p_holm"]),
                r["dz"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["vs", "metric", "N", "mean_a", "std_a", "mean_b", "std_b", "delta",
            "ci_lo", "ci_hi", "shapiro_p", "t_p", "wilcoxon_p", "wilcoxon_p_holm",
            "dz", "cliff", "better"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_rows:
            w.writerow([r.get(c, "") if isinstance(r.get(c), str)
                        else ("%.6g" % r.get(c) if r.get(c) == r.get(c) else "")
                        for c in cols])

    print("\n[out] %s" % args.out)
    print("\n注：*** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著。")
    print("    论文中应报告 Holm 校正后的 p 值，并同时给出效应量 dz —— ")
    print("    N=824 时极小的差异也可能显著，仅报 p 值会被审稿人质疑。")


if __name__ == "__main__":
    main()
