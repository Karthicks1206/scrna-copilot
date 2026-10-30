"""Re-run the full pipeline on PBMC 3k under several random seeds and report how stable the results are.

Usage: python scripts/seed_stability.py --seeds 0 1 2 3 4
Writes results/seed_stability.csv (one row per seed) and prints a summary.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from scrna_copilot.pipeline import PipelineConfig, load_pbmc3k, run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    base = PipelineConfig()
    runs = {}
    for seed in args.seeds:
        adata = load_pbmc3k()
        run_all(adata, replace(base, random_state=seed))
        runs[seed] = adata.obs[["leiden", "cell_type"]].astype(str)
        print(f"seed {seed}: done")

    ref = runs[args.seeds[0]]
    rows = []
    for seed, obs in runs.items():
        counts = obs["cell_type"].value_counts()
        rows.append({
            "seed": seed,
            "n_clusters": obs["leiden"].nunique(),
            "ari_cell_type_vs_first": round(adjusted_rand_score(ref["cell_type"], obs["cell_type"]), 3),
            "ari_clusters_vs_first": round(adjusted_rand_score(ref["leiden"], obs["leiden"]), 3),
            **{f"n_{k}": int(v) for k, v in counts.sort_index().items()},
        })
    df = pd.DataFrame(rows).fillna(0)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seed_stability.csv", index=False)
    print(df.T.to_string(header=False))


if __name__ == "__main__":
    main()
