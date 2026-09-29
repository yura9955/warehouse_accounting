"""Общие фикстуры: справочники, история расхода и словари.

Пути вычисляются от расположения файла, чтобы тесты запускались
из любой рабочей директории.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from normalization import normalizer as nm


_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


@pytest.fixture(scope = "session")
def catalog() -> nm.Catalog:
    """Справочник товаров из data/catalog.json."""
    return nm.Catalog.load(str(_DATA / "catalog.json"))


@pytest.fixture(scope = "session")
def dicts() -> nm.DataDictionaries:
    """Словари вариантов написания из data/dictionaries.json."""
    return nm.DataDictionaries.load(str(_DATA / "dictionaries.json"))


@pytest.fixture(scope = "session")
def history() -> dict:
    """История расхода из data/history.json."""
    with open(_DATA / "history.json", encoding = "utf-8") as f:
        return json.load(f)