"""Fast tests on synthetic count data (no download needed)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from anndata import AnnData

from scrna_copilot.pipeline import PipelineConfig, annotate, load_h5ad, run_all, run_qc


def synthetic_adata(n_cells: int = 300, n_genes: int = 600, seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    # three cell populations, each with its own block of highly expressed genes
    groups = np.repeat([0, 1, 2], n_cells // 3)
    lam = np.full((n_cells, n_genes), 0.5)
    for g in range(3):
        lam[groups == g, g * 50:(g + 1) * 50] = 8.0
    counts = rng.poisson(lam).astype(np.float32)
    var = pd.DataFrame(index=[f"MT-{i}" if i < 5 else f"GENE{i}" for i in range(n_genes)])
    obs = pd.DataFrame({"true_group": groups.astype(str)}, index=[f"cell{i}" for i in range(n_cells)])
    return AnnData(counts, obs=obs, var=var)


@pytest.fixture
def cfg():
    return PipelineConfig(min_genes=50, max_genes=10_000, max_pct_mt=50, n_top_genes=300, n_pcs=20, resolution=0.3)


def test_qc_filters_and_reports(cfg):
    adata = synthetic_adata()
    summary = run_qc(adata, cfg)
    assert summary["cells_after"] <= summary["cells_before"]
    assert "pct_counts_mt" in adata.obs


def test_full_pipeline_recovers_groups(cfg):
    adata = synthetic_adata()
    result = run_all(adata, cfg)
    assert len(result["cluster_sizes"]) >= 3
    assert not result["markers"].empty
    steps = [p["step"] for p in adata.uns["provenance"]]
    assert steps[:2] == ["config", "qc"] and "markers" in steps


def test_pipeline_is_reproducible(cfg):
    a, b = synthetic_adata(), synthetic_adata()
    run_all(a, cfg)
    run_all(b, cfg)
    assert (a.obs["leiden"].values == b.obs["leiden"].values).all()


def test_outputs_roundtrip_with_provenance(cfg, tmp_path):
    adata = synthetic_adata()
    run_all(adata, cfg, outdir=tmp_path)
    assert (tmp_path / "markers.csv").exists()
    reloaded = load_h5ad(tmp_path / "processed.h5ad")
    steps = [p["step"] for p in reloaded.uns["provenance"]]
    assert "qc" in steps and "markers" in steps


SYNTH_SIGNATURES = {
    "type A": {"pos": ["GENE10", "GENE11"], "neg": ["GENE60"]},
    "type B": {"pos": ["GENE60", "GENE61"], "neg": ["GENE10"]},
    "type C": {"pos": ["GENE110", "GENE111"], "neg": []},
}


def test_annotate_labels_match_ground_truth(cfg):
    adata = synthetic_adata()
    result = run_all(adata, cfg)
    annotate(adata, signatures=SYNTH_SIGNATURES)
    truth_to_label = {"0": "type A", "1": "type B", "2": "type C"}
    mapped = adata.obs["true_group"].map(truth_to_label)
    assert (mapped.values == adata.obs["cell_type"].astype(str).values).mean() > 0.95
    assert result["annotations"]  # default signatures run without crashing on unknown genes


def test_annotate_unassigned_when_no_signal(cfg):
    adata = synthetic_adata()
    run_all(adata, cfg)
    annotate(adata, signatures={"nothing": {"pos": ["NOT_A_GENE"], "neg": []}})
    assert set(adata.obs["cell_type"]) == {"Unassigned"}


def test_clusters_stable_across_seeds(cfg):
    """Well-separated synthetic groups should cluster the same way under any seed."""
    labels = []
    for seed in (0, 1, 2):
        adata = synthetic_adata()
        run_all(adata, PipelineConfig(**{**cfg.__dict__, "random_state": seed}))
        labels.append(adata.obs["leiden"].astype(str).values)
    assert adjusted_rand_score(labels[0], labels[1]) > 0.95
    assert adjusted_rand_score(labels[0], labels[2]) > 0.95
