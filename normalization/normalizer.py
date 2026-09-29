from __future__ import annotations
import json
import re
from dataclasses import dataclass
from . import deterministic_parser as dp
from . import regex_parser as rg


def normalize_movement(text: str) -> dict:
    result = {"date": None,
              "sku": None,
              "location": None,
              "operation": None,
              "qty": None,
              "unit": None,
              "batch": None,
              "doc_no": None}

    catalog = Catalog.load("data/catalog.json")
    data_dict = DataDictionaries.load("data/dictionaries.json")

    result = rg.regex(text, result, data_dict)  # Поиск ифнорамации в предлоежние
    if result["sku"] is None:
        result["sku"] = find_sku_by_name(text, catalog)  #Поиск name из справочника
    result = dp.Deterministic_parser(result, data_dict, catalog)  # Преведение в единый вид

    return result

def find_sku_by_name(text: str, catalog: Catalog) -> str :
    toks = {w[:5] for w in re.findall(r"[а-яёa-z]+", text.lower()) if len(w) >= 4}
    counts: dict[str, int] = {}
    for t in toks:
        for sku in catalog.by_token.get(t, ()):
            counts[sku] = counts.get(sku, 0) + 1
    if counts:
        result_sku = max(counts, key=counts.get)
        return result_sku.lower()

@dataclass
class Catalog:
    by_sku: dict[str, dict]
    by_token: dict[str, set[str]]

    @classmethod
    def load(cls, path: str) -> "Catalog":
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        by_sku = {r["sku"]: r for r in rows}

        idx: dict[str, set[str]] = {}
        for sku, item in by_sku.items():
            for w in re.findall(r"[а-яёa-z]+", item["name"].lower()):
                if len(w) >= 4:
                    idx.setdefault(w[:5], set()).add(sku)

        return cls(by_sku=by_sku, by_token=idx)

    def get(self, sku):
        return self.by_sku.get(sku)


class DataDictionaries:
    def __init__(self, months: dict[str, str], sku: dict[str, str],
                 location: dict[str, str], operations: dict[str, str],
                 qty: dict[str, str], units: dict[str, str],
                 batch: dict[str, str], doc_no: dict[str, str], ) -> None:
        self.months = months
        self.sku = sku
        self.location = location
        self.operations = operations
        self.qty = qty
        self.units = units
        self.batch = batch
        self.doc_no = doc_no

    @classmethod
    def load(cls, file_path: str) -> "DataDictionaries":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            months=cls._invert(data.get("months", {})),
            sku=cls._invert(data.get("sku", {})),
            location=cls._invert(data.get("location", {})),
            operations=cls._invert(data.get("operations", {})),
            qty=cls._invert(data.get("qty", {})),
            units=cls._invert(data.get("units", {})),
            batch=cls._invert(data.get("batch", {})),
            doc_no=cls._invert(data.get("doc_no", {})),
        )

    @staticmethod
    def _invert(raw: dict[str, list[str]]) -> dict[str, str]:
        return {variant: key for key, variants in raw.items() for variant in variants}