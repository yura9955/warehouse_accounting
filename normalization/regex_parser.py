"""
Извлечение сырых фрагментов из текста регулярками.
Возвращает то, что найдено без нормализации.
"""
from __future__ import annotations
import re


def regex(text: str, result: dict, dicts_config: DataDictionaries) -> dict:
    clean_text = text.lower()
    result['date'] = regex_data(clean_text, dicts_config.months)
    result['sku'] = regex_sku_location_operation(clean_text, dicts_config.sku)
    result['location'] = regex_sku_location_operation(clean_text, dicts_config.location)
    result['operation'] = regex_sku_location_operation(clean_text, dicts_config.operations)
    result['qty'] = regex_qty_units(clean_text, dicts_config.qty, dicts_config.units)
    result['batch'] = regex_batch_and_doc_no(clean_text, dicts_config.batch)
    result['doc_no'] = regex_batch_and_doc_no(clean_text, dicts_config.doc_no)

    return result


def regex_data(text: str, dicts_config: dict) -> str:
    sorted_months = sorted(dicts_config.keys(), key = len, reverse = True)
    months_part = "|".join(re.escape(k) for k in sorted_months)

    date_pattern = re.compile(r"\b\d{1,4}[-./ ](" + months_part + r"|\d{1,2})[-./ ]\d{2,4}\b")

    match_date = date_pattern.search(text)
    if match_date:
        return match_date.group(0)


def regex_sku_location_operation(text: str, dicts_config: dict) -> str:
    sorted_str = sorted(dicts_config.keys(), key = len, reverse = True)
    fin_str = "|".join(re.escape(s) for s in sorted_str)

    pattern = re.compile(rf"\b({fin_str})\b")

    match = pattern.search(text)
    if match:
        return match.group(0)


def regex_qty_units(text: str, dicts_config_qty: dict, dicts_config_units: dict) -> str:
    sorted_qty = sorted(dicts_config_qty.keys(), key = len, reverse = True)
    qty_str = "|".join(re.escape(u) for u in sorted_qty)

    sorted_units = sorted(dicts_config_units.keys(), key = len, reverse = True)
    units_str = "|".join(re.escape(u) for u in sorted_units)

    qty_pattern = rf"(-?\d+(?:[.,]\d+)?)\s*({units_str})"
    qty_pattern_two = rf"(\d+)\s*({qty_str})\s*(?:по|х|\*)\s*(\d+(?:[.,]\d+)?)\s*({units_str})"

    if match_qty_two := re.search(qty_pattern_two, text):
        return match_qty_two.group(0)
    elif match_qty := re.search(qty_pattern, text):
        return match_qty.group(0)


def regex_batch_and_doc_no(text: str, dicts_config: dict) -> str:
    sorted_batch = sorted(dicts_config.keys(), key = len, reverse = True)
    batch_str = "|".join(re.escape(b) for b in sorted_batch)

    batch_pattern = re.compile(rf"\b(?:{batch_str})[-_ ]?[a-z0-9][a-z0-9-]*\b")

    match_batch = re.search(batch_pattern, text)
    if match_batch:
        return match_batch.group(0).upper()