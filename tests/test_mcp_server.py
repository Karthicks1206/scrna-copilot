"""Tests for the MCP tool layer (called in-process, no Claude Desktop needed)."""
import asyncio

import pytest

from scrna_copilot import mcp_server as srv
from scrna_copilot.pipeline import PipelineConfig
from tests.test_pipeline import synthetic_adata

EXPECTED_TOOLS = {"load_dataset", "quality_control", "normalize_and_pca", "cluster_cells",
                  "marker_genes", "annotate_cell_types", "provenance"}


@pytest.fixture(autouse=True)
def fresh_state():
    srv._state.update(adata=None, cfg=PipelineConfig(), stage="empty")
    yield


def test_server_exposes_expected_tools():
    tools = asyncio.run(srv.mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_tools_refuse_to_run_out_of_order():
    with pytest.raises(ValueError, match="earlier steps"):
        srv.quality_control()


def test_tool_chain_end_to_end(tmp_path):
    path = tmp_path / "synthetic.h5ad"
    synthetic_adata().write_h5ad(path)
    assert srv.load_dataset(str(path))["n_cells"] == 300
    qc = srv.quality_control(min_genes=50, max_genes=10_000, max_pct_mt=50)
    assert qc["cells_after"] > 0
    srv.normalize_and_pca(n_top_genes=300, n_pcs=20)
    clusters = srv.cluster_cells(resolution=0.3)
    assert clusters["n_clusters"] >= 3
    assert srv.marker_genes(top_n=3)
    assert srv.annotate_cell_types()
    steps = [s["step"] for s in srv.provenance()["steps"]]
    assert steps == ["load", "qc", "normalize_reduce", "cluster", "markers", "annotate"]
