from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

import data_manipulation as dm


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Bylaws_2022.pdf", "bylaw"),
        ("by-law_amendment.pdf", "bylaw"),
        ("STRATA_COUnCIL_MEETING_MINUTES.pdf", "minutes"),
        ("AGM_package_2023.pdf", "minutes"),
        ("SGM_2024_notes.md", "minutes"),
        ("notice_rule_change_2021.txt", "notice"),
        ("financial_budget_2020.pdf", "financial"),
        ("random_doc.pdf", "other"),
    ],
)
def test_infer_doc_type(name: str, expected: str):
    assert dm._infer_doc_type(Path(name)) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("minutes_2022_agm.pdf", "2022"),
        ("bylaws_1999.pdf", "1999"),
        ("no_year_here.pdf", None),
        ("weird_20_22.pdf", None),
    ],
)
def test_infer_year(name: str, expected: str | None):
    assert dm._infer_year(Path(name)) == expected


def test_get_data_list_discovers_allowed_extensions(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "bylaws_2022.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "a" / "minutes_2023_agm.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "a" / "ignore.png").write_bytes(b"\x89PNG")

    df = dm.get_data_list(str(tmp_path))
    assert set(df.columns) >= {"Title", "path", "doc_type", "year"}
    assert len(df) == 2
    assert all(Path(p).is_absolute() for p in df["path"].tolist())


def test_get_data_list_missing_folder_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        dm.get_data_list(str(missing))


def test_clean_document_txt_normalizes_whitespace(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("Hello   world\n\nThis\tis  spaced.\n", encoding="utf-8")
    cleaned = dm.clean_document(str(p))
    assert cleaned == "Hello world This is spaced."


def test_clean_document_pdf_uses_loader_and_normalizes(monkeypatch, tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")

    class FakeLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            return [
                Document(page_content="Hello   PDF"),
                Document(page_content="Page 2\n\nMore"),
            ]

    monkeypatch.setattr(dm, "PyPDFLoader", FakeLoader)
    cleaned = dm.clean_document(str(p))
    assert cleaned == "Hello PDF Page 2 More"


def test_create_text_splitter_applies_project_splitter(monkeypatch):
    df = pd.DataFrame([{"Content": "abc"}, {"Content": "def"}])
    calls: list[str] = []

    def fake_splitter(text: str):
        calls.append(text)
        return [text, text + "!"]

    monkeypatch.setattr(dm, "text_splitter", fake_splitter)
    out = dm.create_text_splitter(df)
    assert "chunked_text" in out.columns
    assert out["chunked_text"].tolist() == [["abc", "abc!"], ["def", "def!"]]
    assert calls == ["abc", "def"]

