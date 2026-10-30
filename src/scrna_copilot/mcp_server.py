"""MCP server that lets an LLM agent (e.g. Claude) drive the scRNA-seq pipeline.

Tools are stateful: the loaded dataset stays in memory between calls, so the
agent can run QC, inspect the result, adjust parameters, then cluster.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from scrna_copilot.pipeline import (
    PipelineConfig, _data_dir, annotate, cluster, find_markers, load_h5ad, load_pbmc3k,
    normalize_and_reduce, run_qc,
)

mcp = FastMCP("scrna-copilot")
_state: dict = {"adata": None, "cfg": PipelineConfig(), "stage": "empty"}


def _require(stage: str) -> None:
    order = ["empty", "loaded", "qc", "reduced", "clustered", "annotated"]
    if order.index(_state["stage"]) < order.index(stage):
        raise ValueError(f"Run the earlier steps first (current stage: {_state['stage']}, needs: {stage}).")


def _resolve_source(source: str) -> Path:
    """Only .h5ad files inside the data directory may be loaded, so an agent can't read arbitrary paths."""
    allowed = _data_dir().resolve()
    path = Path(source).expanduser().resolve()
    if path.suffix != ".h5ad" or not path.is_relative_to(allowed):
        raise ValueError(f"Only .h5ad files inside {allowed} can be loaded (got: {source}).")
    return path


@mcp.tool()
def load_dataset(source: str = "pbmc3k") -> dict:
    """Load a dataset. Use 'pbmc3k' for the public 10x PBMC 3k data, or the path of an .h5ad file inside the data directory."""
    adata = load_pbmc3k() if source == "pbmc3k" else load_h5ad(_resolve_source(source))
    _state.update(adata=adata, stage="loaded")
    return {"n_cells": adata.n_obs, "n_genes": adata.n_vars}


@mcp.tool()
def quality_control(min_genes: int = 200, max_genes: int = 2500, max_pct_mt: float = 5.0) -> dict:
    """Filter low-quality cells. Returns cells kept, genes kept, and median QC metrics."""
    _require("loaded")
    cfg = _state["cfg"]
    cfg.min_genes, cfg.max_genes, cfg.max_pct_mt = min_genes, max_genes, max_pct_mt
    summary = run_qc(_state["adata"], cfg)
    _state["stage"] = "qc"
    return summary


@mcp.tool()
def normalize_and_pca(n_top_genes: int = 2000, n_pcs: int = 40) -> dict:
    """Normalize, log-transform, select highly variable genes, scale, and run PCA."""
    _require("qc")
    cfg = _state["cfg"]
    cfg.n_top_genes, cfg.n_pcs = n_top_genes, n_pcs
    normalize_and_reduce(_state["adata"], cfg)
    _state["stage"] = "reduced"
    var_ratio = _state["adata"].uns["pca"]["variance_ratio"][:10]
    return {"top10_pc_variance_ratio": [round(float(v), 4) for v in var_ratio]}


@mcp.tool()
def cluster_cells(resolution: float = 1.0, n_neighbors: int = 10) -> dict:
    """Leiden clustering + UMAP. Higher resolution gives more, smaller clusters."""
    _require("reduced")
    cfg = _state["cfg"]
    cfg.resolution, cfg.n_neighbors = resolution, n_neighbors
    sizes = cluster(_state["adata"], cfg)
    _state["stage"] = "clustered"
    return {"n_clusters": len(sizes), "cluster_sizes": sizes}


@mcp.tool()
def marker_genes(top_n: int = 5) -> dict:
    """Top marker genes per cluster (Wilcoxon). Use these to suggest cell-type labels."""
    _require("clustered")
    df = find_markers(_state["adata"], top_n=top_n)
    return {g: sub["names"].tolist() for g, sub in df.groupby("group", observed=True)}


@mcp.tool()
def annotate_cell_types(min_score: float = 25.0) -> dict:
    """Label each cluster with a PBMC cell type by scoring canonical marker genes. Review against marker_genes."""
    _require("clustered")
    result = annotate(_state["adata"], min_score=min_score)
    _state["stage"] = "annotated"
    return result


@mcp.tool()
def provenance() -> dict:
    """Every step run so far with its parameters, for reproducibility."""
    adata = _state["adata"]
    return {
        "stage": _state["stage"],
        "config": asdict(_state["cfg"]),
        "steps": [] if adata is None else adata.uns.get("provenance", []),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
