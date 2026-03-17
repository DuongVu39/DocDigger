from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

import index_hoa_docs as indexer


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg = {"docs_path": "data/hoa_docs", "vector_store": {"store_name": "s", "index_name": "i"}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = indexer.load_config(str(cfg_path))
    assert loaded == cfg


def test_load_config_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        indexer.load_config(str(tmp_path / "missing.yml"))


def test_main_end_to_end_with_temp_docs(monkeypatch, tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "minutes_2023_agm.txt").write_text("Fees are reviewed annually.", encoding="utf-8")
    (docs_dir / "bylaws_2022.md").write_text("Pets allowed.", encoding="utf-8")

    store_name = str(tmp_path / "store")
    index_name = str(tmp_path / "index.bin")

    cfg = {"docs_path": str(docs_dir), "vector_store": {"store_name": store_name, "index_name": index_name}}
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Patch argv parsing (simulate: --config <path>)
    def fake_parse_args(self):
        return argparse.Namespace(config=str(cfg_path))

    monkeypatch.setattr(indexer.argparse.ArgumentParser, "parse_args", fake_parse_args)

    # Make the pipeline fully offline by stubbing embeddings
    from conftest import DeterministicEmbeddings
    import data_manipulation as dm

    monkeypatch.setattr(dm, "create_text_embedding", lambda: DeterministicEmbeddings(dim=8))

    indexer.main()

    assert (tmp_path / "store.pkl").exists()
    assert (tmp_path / "index.bin").exists()

