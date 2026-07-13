"""Persisted version of the manual verification scenarios for disk I/O.

Covers scenario 4: load_product_source against a missing file and an empty/whitespace
file, both expected to raise DataLoadError. All file-based tests use tmp_path so the
real data/product_source.txt is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service.data_loader import DataLoadError, load_product_source


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"

    with pytest.raises(DataLoadError):
        load_product_source(missing)


@pytest.mark.parametrize("contents", ["", "   ", "  \n\t\n  "])
def test_empty_or_whitespace_file_raises(tmp_path: Path, contents: str) -> None:
    empty_file = tmp_path / "product_source.txt"
    empty_file.write_text(contents, encoding="utf-8")

    with pytest.raises(DataLoadError):
        load_product_source(empty_file)


def test_reads_and_strips(tmp_path: Path) -> None:
    source_file = tmp_path / "product_source.txt"
    source_file.write_text("  hello world  \n", encoding="utf-8")

    assert load_product_source(source_file) == "hello world"
