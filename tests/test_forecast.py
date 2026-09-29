"""Тесты Части 2: прогноз потребности и рекомендация закупки.

Покрывают: заполнение пропусков, оценку среднедневного расхода,
расчёт страхового запаса и точки заказа, **округление рекомендованного
объёма до упаковки** (обязательное требование ТЗ), дату исчерпания,
компоненты уверенности и сквозной расчёт forecast_demand.
"""
from __future__ import annotations

import pytest

from forecasts import forecast as fc


class TestFillMissing:
    """Линейная интерполяция пропущенных недель в истории расхода."""

    def test_middle_gap(self):
        # Пропуск в середине (кейс WRAP-030): линейная интерполяция между 6.0 и 6.3
        result = fc.fill_missing([6.2, 5.8, 6.5, 6.0, None, 6.3, 6.1])
        assert result == pytest.approx([6.2, 5.8, 6.5, 6.0, 6.15, 6.3, 6.1])

    def test_no_gap_unchanged(self):
        data = [1.0, 2.0, 3.0]
        assert fc.fill_missing(data) == pytest.approx(data)


class TestAvgDailyConsumption:
    """Оценка среднедневного расхода через EMA по недельной истории."""

    def test_constant(self):
        # Ряд одинаковых значений → EMA равна значению, /7
        assert fc.avg_daily_consumption_estimation([7.0, 7.0, 7.0], 0.5) == pytest.approx(1.0)

    def test_single_value(self):
        assert fc.avg_daily_consumption_estimation([14.0], 0.5) == pytest.approx(2.0)


class TestStockMetrics:
    """Простые формулы страхового запаса и точки заказа."""

    def test_safety_stock(self):
        # Расход 2/день, страховой запас на 10 дней → 20
        assert fc.safety_stock_estimation(2.0, 10) == pytest.approx(20.0)

    def test_reorder_point(self):
        # Страховой запас 20 + расход за lead time 2 * 5 = 30
        assert fc.reorder_point_estimation(20.0, 2.0, 5) == pytest.approx(30.0)


class TestRecommendedQty:
    """Расчёт рекомендованного объёма закупки с округлением до упаковки.

    Проверяем 3 сценария из ТЗ:
      1) заказ не нужен (запас с запасом выше точки заказа);
      2) заказ нужен, значение уже кратно упаковке;
      3) заказ нужен, значение округляется вверх до упаковки.
    """

    @pytest.mark.parametrize("stock, incoming, reorder, forecast, pack, expected", [
        (100, 0, 20, 30, 5, 0),       # доступный запас сильно выше точки заказа
        (50, 10, 30, 20, 5, 0),       # available − forecast > reorder
        (20, 0, 30, 5, 5, 15),        # ровно 15, кратно pack_size
        (22, 0, 30, 0, 5, 10),        # 8 → вверх до 10 (кратно 5)
        (5, 0, 30, 100, 5, 125),      # отрицательный «запас» → большой заказ
        (25, 0, 30, 0, 1, 5),         # pack_size = 1, округление не влияет
        (0, 0, 10, 0, 3, 12),         # 10 → вверх до 12 (кратно 3)
    ])
    def test_cases(self, stock, incoming, reorder, forecast, pack, expected):
        result = fc.recommended_qty_estimation(stock, incoming, reorder, forecast, pack)
        assert result == expected


class TestEstimatedCost:
    """Стоимость заказа по цене из справочника."""

    def test_cost(self):
        assert fc.estimated_cost_estimation(10, 100.0) == pytest.approx(1000.0)

    def test_zero_qty(self):
        assert fc.estimated_cost_estimation(0, 100.0) == pytest.approx(0.0)


class TestStockoutDate:
    """Дата исчерпания запаса при текущем расходе."""

    def test_normal(self):
        # 70 единиц при расходе 1/день → +70 дней от 2026-09-15
        assert fc.stockout_date_estimation(70.0, 0.0, 1.0, "2026-09-15") == "2026-11-24"

    def test_zero_avg_daily(self):
        # Расхода нет — дата исчерпания не определена, возвращаем as_of
        assert fc.stockout_date_estimation(50.0, 0.0, 0.0, "2026-09-15") == "2026-09-15"

    def test_empty_stock(self):
        # Запас нулевой — исчерпание уже наступило
        assert fc.stockout_date_estimation(0.0, 0.0, 1.0, "2026-09-15") == "2026-09-15"


class TestConfidenceComponents:
    """Компоненты уверенности прогноза."""

    def test_history_empty(self):
        assert fc._confidence_history([]) == pytest.approx(0.0)

    def test_history_full(self):
        # 52 недели → 1.0
        assert fc._confidence_history([1.0] * 52) == pytest.approx(1.0)

    def test_horizon_zero(self):
        assert fc._confidence_horizon(0) == pytest.approx(1.0)

    def test_horizon_90d(self):
        # 90 дней → 0.5 (формула 1/(1 + h/90))
        assert fc._confidence_horizon(90) == pytest.approx(0.5)

    def test_volatility_constant_series(self):
        # Разброса нет → уверенность 1.0
        assert fc._confidence_volatility([5.0] * 10) == pytest.approx(1.0)


class TestForecastDemandEndToEnd:
    """Сквозной расчёт forecast_demand на данных из Приложения 1."""

    PARAMS = {"method": "ema", "weights_ema": 0.5, "threshold": 0.65}

    def test_oil_001_90d(self, history, catalog):
        result = fc.forecast_demand(history, "OIL-001", 90, catalog, self.PARAMS)
        assert result["avg_daily_consumption"] == pytest.approx(1.83, abs=0.01)
        assert result["recommended_qty"] == 135
        assert result["estimated_cost"] == pytest.approx(169971.75)

    def test_oil_001_30d(self, history, catalog):
        result = fc.forecast_demand(history, "OIL-001", 30, catalog, self.PARAMS)
        assert result["recommended_qty"] == 25
        assert result["estimated_cost"] == pytest.approx(31476.25)

    def test_scrb_020_180d(self, history, catalog):
        result = fc.forecast_demand(history, "SCRB-020", 180, catalog, self.PARAMS)
        assert result["recommended_qty"] == 199
        assert result["estimated_cost"] == pytest.approx(356110.50)

    def test_unknown_sku_returns_empty(self, history, catalog):
        result = fc.forecast_demand(history, "UNKNOWN", 30, catalog, self.PARAMS)
        assert result["avg_daily_consumption"] is None
        assert result["recommended_qty"] is None