"""Тесты Части 1: нормализация записей движения товара.

Покрывают: разбор дат (4 формата из ТЗ), разбор и приведение количества
к базовой единице справочника, поиск SKU по названию, сквозной прогон
normalize_movement на записях M1–M2 из Приложения 1.
"""
from __future__ import annotations

import pytest

from normalization import deterministic_parser as dp
from normalization import normalizer as nm


class TestParserDate:
    """Разбор даты из текста в ISO-формат YYYY-MM-DD."""

    @pytest.mark.parametrize("raw, expected", [
        ("05.03.2026", "2026-03-05"),       # DD.MM.YYYY
        ("05.03.26", "2026-03-05"),          # DD.MM.YY
        ("2026-03-08", "2026-03-08"),        # ISO (YYYY-MM-DD)
        ("2026.03.08", "2026-03-08"),        # YYYY.MM.DD
        ("03/06/26", "2026-06-03"),          # DD/MM/YY
        ("1 марта 2026", "2026-03-01"),      # словесный месяц
        ("12 марта 2026 г.", "2026-03-12"),  # словесный месяц с «г.»
    ])
    def test_formats(self, raw, expected, dicts):
        assert dp.parser_date(raw, dicts.months) == expected


class TestParserQty:
    """Разбор количества и приведение к базовой единице из справочника."""

    def test_ml_to_l(self, catalog):
        # OIL-001: базовая единица — литр, вход — миллилитры
        assert dp.parser_qty("450 мл", "OIL-001", catalog) == pytest.approx(0.45)

    def test_kg_direct(self, catalog):
        # SCRB-020: базовая единица — кг, вход — кг (без конверсии)
        assert dp.parser_qty("3,5 кг", "SCRB-020", catalog) == pytest.approx(3.5)

    def test_pack_multiplier_l(self, catalog):
        # «2 канистры по 5 л» → 10 л
        assert dp.parser_qty("2 канистры по 5 л", "OIL-001", catalog) == pytest.approx(10.0)

    def test_pack_multiplier_pairs(self, catalog):
        # «4 уп. по 50 пар» → 200 пар (CONS-051)
        assert dp.parser_qty("4 уп. по 50 пар", "CONS-051", catalog) == pytest.approx(200.0)

    def test_negative_correction(self, catalog):
        # Корректировка может быть отрицательной (M7)
        assert dp.parser_qty("−120 шт", "CONS-052", catalog) == pytest.approx(-120.0)

    def test_empty_returns_none(self, catalog):
        assert dp.parser_qty(None, "OIL-001", catalog) is None
        assert dp.parser_qty("", "OIL-001", catalog) is None


class TestParserOperation:
    """Приведение операций к значениям receipt / consume / writeoff / return / correction."""

    class TestParserOperation:
        """Приведение операций к receipt / consume / writeoff / return / correction.

        В dictionaries.json операции хранятся как стемы-варианты («списан»,
        «коррект»), поэтому тест подаёт именно те строки, которые реально
        окажутся в словаре после инверсии в DataDictionaries.load.
        """

        @pytest.mark.parametrize("raw, expected", [
            ("приход", "receipt"),
            ("расход", "consume"),
            ("списан", "writeoff"),
            ("возврат", "return"),
            ("коррект", "correction"),
        ])
        def test_operations(self, raw, expected, dicts):
            assert dp.parser_operation(raw, dicts.operations) == expected

class TestFindSkuByName:
    """Поиск SKU по названию, когда явный код в тексте не указан."""

    def test_cons_051_by_name(self, catalog):
        # «тапочки одноразовые» — название CONS-051
        found = nm.find_sku_by_name("Приход тапочек одноразовых", catalog)
        assert found is not None
        assert found.upper() == "CONS-051"

    def test_no_match(self, catalog):
        assert nm.find_sku_by_name("что-то совсем непонятное", catalog) is None


class TestNormalizeMovementEndToEnd:
    """Сквозной прогон normalize_movement на записях M1–M2 из Приложения 1.

    Функция подгружает справочник и словари сама (сигнатура из ТЗ:
    normalize_movement(text) -> dict), поэтому фикстуры не нужны.
    """

    def test_m1(self):
        text = "05.03.2026 MS-01 приход OIL-001, 2 канистры по 5 л, НК-345"
        result = nm.normalize_movement(text)
        assert result["date"] == "2026-03-05"
        assert result["sku"] == "OIL-001"
        assert result["location"] == "MS-01"
        assert result["operation"] == "receipt"
        assert result["qty"] == pytest.approx(10.0)
        assert result["unit"] == "л"
        assert result["doc_no"] == "НК-345"

    def test_m2(self):
        text = "1 марта 2026 MS-01 расход oil 001 — 450 мл, B-OIL-001-012"
        result = nm.normalize_movement(text)
        assert result["date"] == "2026-03-01"
        assert result["sku"] == "OIL-001"
        assert result["location"] == "MS-01"
        assert result["operation"] == "consume"
        assert result["qty"] == pytest.approx(0.45)
        assert result["unit"] == "л"
        assert result["batch"] == "B-OIL-001-012"