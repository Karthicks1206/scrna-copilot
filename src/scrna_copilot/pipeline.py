"""Reproducible single-cell RNA-seq analysis pipeline (scanpy).

Steps: load -> QC -> normalize -> highly variable genes -> PCA ->
neighbors -> Leiden clustering -> UMAP -> marker genes.
Every step records its parameters in adata.uns["provenance"] so a run
can be audited and reproduced.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


@dataclass
class PipelineConfig:
    min_genes: int = 200          # drop cells with fewer detected genes
    min_cells: int = 3            # drop genes seen in fewer cells
    max_genes: int = 2500         # drop likely doublets
    max_pct_mt: float = 5.0       # drop cells with high mitochondrial reads
    target_sum: float = 1e4       # counts per cell after normalization
    n_top_genes: int = 2000       # highly variable genes kept
    n_pcs: int = 40
    n_neighbors: int = 10
    resolution: float = 1.0       # Leiden resolution (higher = more clusters); 1.0 recovers NK cells in 10/10 seeds on PBMC 3k
    random_state: int = 0

# Marker signatures for PBMC cell types. "pos" genes should be expressed in the
# population, "neg" genes should not. Chosen from canonical lineage markers and
# checked against the PBMC 3k run (see README).
PBMC_SIGNATURES: dict[str, dict[str, list[str]]] = {
    "CD4+ T cells": {"pos": ["CD3D", "CD3E", "IL7R"], "neg": ["NKG7"]},
    "CD8+ T cells": {"pos": ["CD3D", "CD8A", "GZMK", "NKG7"], "neg": []},
    "NK cells": {"pos": ["GNLY", "NKG7", "GZMB"], "neg": ["CD3D", "CD3E"]},
    "B cells": {"pos": ["MS4A1", "CD79A"], "neg": ["CD3D"]},
    "CD14+ monocytes": {"pos": ["CD14", "LYZ", "CST3"], "neg": ["FCGR3A", "FCER1A"]},
    "FCGR3A+ monocytes": {"pos": ["FCGR3A", "MS4A7", "LYZ", "CST3"], "neg": ["CD14", "FCER1A"]},
    "Dendritic cells": {"pos": ["FCER1A", "CLEC10A", "CST3"], "neg": ["CD3D"]},
    "Platelets": {"pos": ["PPBP", "PF4"], "neg": []},
}


def _log(adata: AnnData, step: str, **params) -> None:
    adata.uns.setdefault("provenance", []).append({"step": step, **params})


def _data_dir() -> Path:
    """Dataset cache: $SCRNA_COPILOT_DATA_DIR, else <repo>/data, else ~/.cache/scrna-copilot.

    Anchored to a fixed path so it doesn't depend on the working directory (an MCP
    client such as Claude Desktop launches the server from an arbitrary directory).
    """
    if os.environ.get("SCRNA_COPILOT_DATA_DIR"):
        return Path(os.environ["SCRNA_COPILOT_DATA_DIR"])
    repo = Path(__file__).resolve().parents[2]
    return repo / "data" if (repo / "pyproject.toml").exists() else Path.home() / ".cache" / "scrna-copilot"


def load_pbmc3k() -> AnnData:
    """Public 10x Genomics PBMC 3k dataset (downloads on first use)."""
    sc.settings.datasetdir = _data_dir()
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    _log(adata, "load", source="10x Genomics PBMC 3k", n_cells=adata.n_obs, n_genes=adata.n_vars)
    return adata


def load_h5ad(path: str | Path) -> AnnData:
    adata = sc.read_h5ad(path)
    if "provenance_json" in adata.uns:  # restore history saved by save_h5ad
        adata.uns["provenance"] = json.loads(adata.uns.pop("provenance_json"))
    adata.var_names_make_unique()
    _log(adata, "load", source=str(path), n_cells=adata.n_obs, n_genes=adata.n_vars)
    return adata


def save_h5ad(adata: AnnData, path: str | Path) -> None:
    """Write to .h5ad. h5py can't store a list of dicts, so provenance is saved as JSON."""
    prov = adata.uns.pop("provenance", [])
    adata.uns["provenance_json"] = json.dumps(prov, default=str)
    try:
        adata.write_h5ad(path)
    finally:
        adata.uns.pop("provenance_json", None)
        adata.uns["provenance"] = prov


def run_qc(adata: AnnData, cfg: PipelineConfig) -> dict:
    """Filter low-quality cells and genes. Returns a QC summary."""
    before = adata.n_obs
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)
    keep = (adata.obs["n_genes_by_counts"] < cfg.max_genes) & (adata.obs["pct_counts_mt"] < cfg.max_pct_mt)
    adata._inplace_subset_obs(keep.values)
    summary = {
        "cells_before": int(before),
        "cells_after": int(adata.n_obs),
        "genes_after": int(adata.n_vars),
        "median_genes_per_cell": float(adata.obs["n_genes_by_counts"].median()),
        "median_pct_mt": float(adata.obs["pct_counts_mt"].median()),
    }
    _log(adata, "qc", min_genes=cfg.min_genes, min_cells=cfg.min_cells,
         max_genes=cfg.max_genes, max_pct_mt=cfg.max_pct_mt, **summary)
    return summary


def normalize_and_reduce(adata: AnnData, cfg: PipelineConfig) -> None:
    """Normalize, log-transform, select HVGs, scale, PCA."""
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full log-normalized matrix for marker genes
    n_top = min(cfg.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    adata._inplace_subset_var(adata.var["highly_variable"].values)
    sc.pp.scale(adata, max_value=10)
    n_pcs = min(cfg.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_pcs, random_state=cfg.random_state)
    _log(adata, "normalize_reduce", target_sum=cfg.target_sum, n_top_genes=n_top, n_pcs=n_pcs)


def cluster(adata: AnnData, cfg: PipelineConfig) -> dict:
    """Build kNN graph, run Leiden clustering and UMAP. Returns cluster sizes."""
    n_pcs = adata.obsm["X_pca"].shape[1]
    sc.pp.neighbors(adata, n_neighbors=cfg.n_neighbors, n_pcs=n_pcs, random_state=cfg.random_state)
    sc.tl.leiden(adata, resolution=cfg.resolution, random_state=cfg.random_state,
                 flavor="igraph", n_iterations=-1, directed=False)
    sc.tl.umap(adata, random_state=cfg.random_state)
    sizes = adata.obs["leiden"].value_counts().sort_index().to_dict()
    sizes = {str(k): int(v) for k, v in sizes.items()}
    _log(adata, "cluster", n_neighbors=cfg.n_neighbors, resolution=cfg.resolution, n_clusters=len(sizes))
    return sizes


def find_markers(adata: AnnData, top_n: int = 5) -> pd.DataFrame:
    """Top marker genes per cluster (Wilcoxon rank-sum test)."""
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    top = df.groupby("group", observed=True).head(top_n).reset_index(drop=True)
    _log(adata, "markers", method="wilcoxon", top_n=top_n)
    return top


def annotate(adata: AnnData, signatures: dict | None = None, groupby: str = "leiden",
             min_score: float = 25.0) -> dict:
    """Label each cluster with the best-matching cell-type signature.

    Score = mean % of cells expressing the signature's "pos" genes minus the mean %
    expressing its "neg" genes. Clusters whose best score is below `min_score` are
    labelled "Unassigned". Labels are written to adata.obs["cell_type"].
    """
    sigs = signatures or PBMC_SIGNATURES
    src = adata.raw if adata.raw is not None else adata
    wanted = sorted({g for sig in sigs.values() for g in sig["pos"] + sig["neg"]})
    genes = [g for g in wanted if g in src.var_names]
    X = src[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    clusters = adata.obs[groupby].astype(str).values
    pct = pd.DataFrame(X > 0, columns=genes).groupby(clusters).mean() * 100

    result = {}
    for cl in pct.index:
        best_label, best_score = "Unassigned", float("-inf")
        for label, sig in sigs.items():
            pos = [g for g in sig["pos"] if g in genes]
            neg = [g for g in sig["neg"] if g in genes]
            if not pos:
                continue
            score = pct.loc[cl, pos].mean() - (pct.loc[cl, neg].mean() if neg else 0.0)
            if score > best_score:
                best_label, best_score = label, float(score)
        if best_score < min_score:
            best_label = "Unassigned"
        result[cl] = {"label": best_label, "score": round(best_score, 1),
                      "n_cells": int((clusters == cl).sum())}
    adata.obs["cell_type"] = pd.Categorical(pd.Series(clusters).map(
        {cl: r["label"] for cl, r in result.items()}).values)
    _log(adata, "annotate", method="signature score = %expressing(pos) - %expressing(neg)",
         min_score=min_score, labels={cl: r["label"] for cl, r in result.items()})
    return result


def run_all(adata: AnnData, cfg: PipelineConfig | None = None, outdir: str | Path | None = None) -> dict:
    """Run the full pipeline and optionally save outputs."""
    cfg = cfg or PipelineConfig()
    _log(adata, "config", **asdict(cfg))
    qc = run_qc(adata, cfg)
    normalize_and_reduce(adata, cfg)
    sizes = cluster(adata, cfg)
    markers = find_markers(adata)
    annotations = annotate(adata)
    if outdir:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        markers.to_csv(out / "markers.csv", index=False)
        pd.DataFrame.from_dict(annotations, orient="index").rename_axis("cluster").to_csv(out / "annotations.csv")
        save_h5ad(adata, out / "processed.h5ad")
        sc.settings.figdir = str(out)
        sc.pl.umap(adata, color="leiden", save="_leiden.png", show=False)
        sc.pl.umap(adata, color="cell_type", save="_cell_type.png", show=False)
    return {"qc": qc, "cluster_sizes": sizes, "markers": markers, "annotations": annotations}


if __name__ == "__main__":
    result = run_all(load_pbmc3k(), outdir="results")
    print("QC:", result["qc"])
    print("Clusters:", result["cluster_sizes"])
    print("Annotations:", {k: (v["label"], v["n_cells"]) for k, v in result["annotations"].items()})
    print(result["markers"].head(20).to_string(index=False))
