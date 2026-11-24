"""Reproduce every number in outputs/results.json:  python run_experiments.py"""
from pathlib import Path

from src.experiments import run_all

if __name__ == "__main__":
    results = run_all(Path(__file__).parent)
    c = results["classification"]
    print("CV (train only), top 5 configs:")
    for r in c["cv_table"][:5]:
        print(f"  {r['model']:<26} ngram {r['ngram']}  macroF1 {r['cv_macro_f1']:.3f} ± {r['cv_macro_f1_sd']:.3f}")
    h = c["holdout"]
    print(f"Hold-out ({h['model']}, {h['ngram']}): acc {h['accuracy']:.3f}, macroF1 {h['macro_f1']:.3f}, n={h['n_test']}")
    k = results["clustering"]
    print(f"K-Means k={k['k_used']}: ARI {k['ARI']:.3f}  NMI {k['NMI']:.3f}  purity {k['purity']:.3f}")
    r = results["retrieval"]
    print("Selected retrieval config (dev):", r["selected_on_dev"])
    for row in r["table"]:
        if row["config"] == r["selected_on_dev"]:
            print("  ", row["split"], {m: row[m] for m in ["P@5", "MRR", "nDCG@10"]})
    print("Baseline:", r["keyword_overlap_baseline"])
    print("Failures:", len(r["failures"]), "queries with P@5 < 0.8")
    print("Wrote outputs/results.json")
