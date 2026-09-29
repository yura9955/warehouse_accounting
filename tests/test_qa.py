"""Тесты Части 3: разбор вопроса и формирование ответа.

Покрывают: классификатор интентов (включая стоп-фразы), извлечение
периода / бюджета / SKU / локации, fallback на правилах (без LLM)
и сквозной вызов answer_question.
"""
from __future__ import annotations

import pytest

from aswers_to_questions.answer_question import answer_question
from aswers_to_questions.entity_extractor import (
    extract_budget,
    extract_location,
    extract_period,
    extract_sku,
)
from aswers_to_questions.intent_classifier import classify_intent


class TestClassifyIntent:
    """Определение интента вопроса по ключевым фразам."""

    @pytest.mark.parametrize("question, expected", [
        ("Сколько масла закупить на три месяца?", "forecast_purchase"),
        ("Что нужно заказать в ближайшие 14 дней?", "reorder_list"),
        ("Какой бюджет закупок на квартал?", "budget"),
        ("Что закончится до следующей поставки?", "deficit_risk"),
        ("Какие партии сгорят в этом месяце?", "expiry_risk"),
        ("Насколько подорожало масло?", "price_dynamics"),
        ("Посчитай закупку скраба на полгода при лимите 200 тысяч", "forecast_purchase"),
    ])
    def test_recognised_intents(self, question, expected):
        intent, _, _ = classify_intent(question)
        assert intent == expected

    @pytest.mark.parametrize("question", [
        "Почему план закупок вырос?",       # стоп-фраза «почему»
        "Какая погода в Сочи на выходных?", # стоп-фразы «погода», «на выходных»
        "Когда привезут заказ?",            # стоп-фраза «когда привезут»
    ])
    def test_unknown(self, question):
        intent, _, _ = classify_intent(question)
        assert intent == "unknown"


class TestExtractPeriod:
    """Преобразование периода из вопроса в количество дней."""

    @pytest.mark.parametrize("question, expected", [
        ("на три месяца", 90),
        ("на полгода", 180),
        ("на квартал", 90),
        ("на год", 365),
        ("на 14 дней", 14),
        ("на 2 недели", 14),
        ("на 3 месяца", 90),
        ("в этом месяце", 30),
    ])
    def test_cases(self, question, expected):
        assert extract_period(question) == expected

    def test_no_period(self):
        assert extract_period("что заказать") is None


class TestExtractBudget:
    """Извлечение бюджетного лимита из вопроса."""

    @pytest.mark.parametrize("question, expected", [
        ("при лимите 200 тысяч", 200000.0),
        ("при лимите 1.5 млн", 1500000.0),
        ("в пределах 100000 руб", 100000.0),
        ("лимит 500000", 500000.0),
    ])
    def test_cases(self, question, expected):
        assert extract_budget(question) == pytest.approx(expected)

    def test_no_budget(self):
        assert extract_budget("сколько закупить") is None


class TestExtractSku:
    """Извлечение SKU — по явному коду и по названию."""

    def test_by_code(self, catalog):
        assert extract_sku("сколько oil-001 на складе", catalog) == "OIL-001"

    def test_by_code_with_space(self, catalog):
        assert extract_sku("посчитай scrb 020 на квартал", catalog) == "SCRB-020"

    def test_no_sku(self, catalog):
        assert extract_sku("что заказать", catalog) is None


class TestExtractLocation:
    """Стем-матчинг локаций из dictionaries.json."""

    def test_sochi(self):
        assert extract_location("что закончится в сочи") == "Сочи"

    def test_krasnaya_polyana_inflected(self):
        # Падежная форма «в красной поляне» должна находиться через стемы
        assert extract_location("сколько маски в красной поляне") == "Красная поляна"

    def test_ms_01(self):
        assert extract_location("что на ms-01") == "MS-01"

    def test_no_location(self):
        assert extract_location("сколько масла") is None


class TestAnswerQuestionFallback:
    """Fallback без LLM: разбор через правила + расчёт."""

    @pytest.fixture(autouse=True)
    def disable_llm(self, monkeypatch):
        """Форсируем fallback на правилах — тесты не зависят от API-ключа."""
        monkeypatch.setattr(
            "aswers_to_questions.answer_question.is_llm_available",
            lambda: False,
        )

    def test_forecast_purchase(self, catalog, history):
        parsed, conf, text = answer_question(
            "Сколько масла закупить на три месяца и сколько это будет стоить?",
            context = {"history": history, "catalog": catalog},
        )
        assert parsed["intent"] == "forecast_purchase"
        assert parsed["sku"] == "OIL-001"
        assert parsed["period_days"] == 90
        assert "169971.75" in text

    def test_unknown_when_no_period(self, catalog, history):
        # В вопросе есть SKU, но нет периода → обязательный параметр отсутствует
        parsed, conf, text = answer_question(
            "Заказать масло",
            context = {"history": history, "catalog": catalog},
        )
        assert parsed["intent"] == "unknown"
        assert conf == 0.0
        assert "период" in text.lower()

    def test_stop_phrase_out_of_domain(self, catalog, history):
        # «погода» — стоп-фраза, вопрос вне домена
        parsed, conf, _ = answer_question(
            "Какая погода в Сочи на выходных?",
            context = {"history": history, "catalog": catalog},
        )
        assert parsed["intent"] == "unknown"
        assert conf == 0.0

    def test_missing_catalog_in_context(self):
        # Контекст без справочника — функция не должна падать
        parsed, conf, text = answer_question("что-то", context={})
        assert parsed["intent"] == "unknown"
        assert conf == 0.0
        assert "справочник" in text.lower()