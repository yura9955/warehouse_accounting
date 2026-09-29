import numpy as np
import pandas as pd
from . import config


def forecast_demand(history: dict, sku: str, horizon_days: int, catalog: dict, params: dict | None = None) -> dict:
    """Считает прогноз потребности и рекомендацию по закупке для одной позиции.

    Пайплайн:
        1. Заполняет пропуски в недельной истории (None → линейная интерполяция).
        2. Оценивает среднедневной расход (EMA) и прогноз на горизонт.
        3. Читает текущий остаток и товар в пути.
        4. Считает страховой запас и точку заказа.
        5. Подбирает рекомендуемый объём закупки с округлением до упаковки
           и стоимость по цене из справочника.
        6. Оценивает дату исчерпания запаса.
        7. Считает уверенность прогноза и собирает текстовое объяснение.

    Args:
        history: данные по остаткам, поставкам и недельному расходу на дату as_of.
        sku: код позиции из справочника.
        horizon_days: горизонт прогноза в днях.
        catalog: справочник товаров (unit, pack_size, price, safety_stock_days,
                 lead_time_days и т.д.).
        params: настройки расчёта — weights_ema (вес последней недели EMA),
                method ("ema" / "linear" / "auto"), threshold (порог confidence).

    Returns:
        Словарь с полями: avg_daily_consumption, forecast_demand, current_stock,
        incoming_qty, safety_stock, reorder_point, recommended_qty,
        estimated_cost, stockout_date, confidence, explanation.
        При history=None возвращает словарь с None-полями.
    """

    result = {"avg_daily_consumption": None,  # Дневной расход материла(продукта)
              "forecast_demand": None,  # Расход за весь период, на который строим прогноз
              "current_stock": None,  # Сколько на складе материла(продукта)
              "incoming_qty": None,  # Сколько товара уже едит (Товар не на складе)
              "safety_stock": None,  # Страховой запас
              "reorder_point": None,  # Порог, при котором пора заказывать
              "recommended_qty": None,  # Сколько заказать материла(продукта)
              "estimated_cost": None,  # Стоимоть заказа
              "stockout_date": None,  # Ориентировочная дата окончание материла(продукта)
              "confidence": None,  # Уроверь доверия прогноза
              "explanation": None  # Объяснение из чего собрано число
              }

    if params is None:
        params = {"method": "auto", "weights_ema": 0.5, "threshold": 0.65}

    if history is None:
        return result

    if sku not in history.get("weekly_consumption", {}):
        result["explanation"] = f"{sku}: нет данных о расходе в истории."
        return result

    weeks = fill_missing(weeks=history["weekly_consumption"][sku])

    result["avg_daily_consumption"] = avg_daily_consumption_estimation(weeks=weeks,
                                                                       weights_ema=params["weights_ema"])

    result["forecast_demand"] = forecast_demand_estimation(weeks=weeks,
                                                           horizon_days=horizon_days,
                                                           method=params["method"],
                                                           avg_daily=result["avg_daily_consumption"])

    result["current_stock"] = history["current_stock"][sku]

    result["incoming_qty"] = history["incoming_qty"][sku]

    result["safety_stock"] = safety_stock_estimation(avg_daily_consumption=result["avg_daily_consumption"],
                                                     avg_daily_days=catalog.get(sku)["safety_stock_days"])

    result["reorder_point"] = reorder_point_estimation(safety_stock=result["safety_stock"],
                                                       avg_daily=result["avg_daily_consumption"],
                                                       lead_time=catalog.get(sku)["lead_time_days"])

    result["recommended_qty"] = recommended_qty_estimation(current_stock=result["current_stock"],
                                                           incoming_qty=result["incoming_qty"],
                                                           reorder_point=result['reorder_point'],
                                                           forecast_demand=result['forecast_demand'],
                                                           pack_size=catalog.get(sku)["pack_size"])

    result["estimated_cost"] = estimated_cost_estimation(recommended_qty=result["recommended_qty"],
                                                         price=catalog.get(sku)["price"])

    result["stockout_date"] = stockout_date_estimation(current_stock=result["current_stock"],
                                                       incoming_qty=result["incoming_qty"],
                                                       avg_daily=result["avg_daily_consumption"],
                                                       as_of=history["as_of"])

    result["confidence"], components = confidence_estimation(weeks=weeks,
                                                             horizon_days=horizon_days,
                                                             weights=config.WEIGHTS)

    result["explanation"] = explanation_estimation(result=result,
                                                   sku=sku,
                                                   horizon_days=horizon_days,
                                                   item=catalog.get(sku),
                                                   components=components,
                                                   threshold=params["threshold"])

    return result


def fill_missing(weeks: list) -> list:
    """Заполняет пропуски в недельной истории линейной интерполяцией.

    Args:
        weeks: недельный расход в базовых единицах с пропусками.

    Returns:
        Список той же длины без пропусков.
    """

    arr = np.array([np.nan if w is None else w for w in weeks], dtype=float)
    mask = np.isnan(arr)
    if mask.any():
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr.tolist()


def avg_daily_consumption_estimation(weeks: list[float], weights_ema: float = 0.5) -> float:
    """Оценивает среднедневной расход через экспоненциальное сглаживание недельной истории.

    Args:
        weeks: недельный расход в базовых единицах.
        weights_ema: вес последней недели (0..1). Чем выше, тем сильнее учитывается свежий спрос.

    Returns:
        Среднедневной расход (недельное значение / 7).
    """

    result = weeks[0]
    for week in weeks[1:]:
        result = weights_ema * week + (1 - weights_ema) * result
    return result / 7.0


def forecast_demand_estimation(weeks: list, horizon_days, avg_daily: float, method: str = "auto") -> float:
    """Прогнозирует расход за горизонт в днях по недельной истории.

    Args:
        weeks: недельный расход в базовых единицах.
        horizon_days: горизонт прогноза в днях.
        avg_daily: среднедневной расход (EMA), используется для методов "ema" и "auto".
        method: метод прогноза — "ema", "linear" или "auto".

    Returns:
        Прогнозируемый расход за horizon_days.

    Raises:
        ValueError: если передан неизвестный method.
    """

    if method == "ema":
        result = avg_daily
    elif method == "linear":
        result = _linear_seasonal_est(weeks, horizon_days)
    elif method == "auto":
        result = _auto_est(weeks, horizon_days, avg_daily)
    else:
        raise ValueError(f"unknown method: {method}")

    return result * horizon_days


def _linear_seasonal_est(weeks: list, horizon_days: int, period: int = config.PERIOD) -> float:
    """Оценивает среднедневной расход линейной регрессией с учётом сезонности.

    Если истории меньше периода сезонности (n < period), сезонность не выделяется —
    строится обычная линейная регрессия по тренду. Иначе:
        1. Вычисляется сезонная компонента как отклонение среднего по каждой фазе
           от общего среднего, с центрированием к нулю.
        2. Из ряда вычитается сезонность, по остатку строится линейный тренд.
        3. Прогноз в середине горизонта = тренд + сезонная компонента нужной фазы.

    Args:
        weeks: недельный расход в базовых единицах.
        horizon_days: горизонт прогноза в днях.
        period: длина сезонного цикла в неделях (по умолчанию 12), задаеться в файле config.py.

    Returns:
        Среднедневной расход (прогноз недельного значения / 7).
    """

    y = np.asarray(weeks, dtype=float)
    n = len(y)
    t = np.arange(n)

    if n < period:
        b, a = np.polyfit(t, y, 1)
        t_mid = n - 1 + (horizon_days / 7) / 2
        return (a + b * t_mid) / 7

    phases = t % period
    overall = y.mean()
    season = np.zeros(period)
    filled = []
    for p in range(period):
        m = phases == p
        if m.any():
            season[p] = y[m].mean() - overall
            filled.append(p)

    if filled:
        offset = season[filled].mean()
        season[filled] -= offset

    deseason = y - season[phases]
    b, a = np.polyfit(t, deseason, 1)

    t_mid = n - 1 + (horizon_days / 7) / 2
    return (a + b * t_mid + season[int(t_mid) % period]) / 7


def _auto_est(weeks: list, horizon_days, avg_daily: float, min_weeks: int = config.MIN_WEEKS) -> float:
    """Выбирает метод прогноза автоматически в зависимости от длины истории.

    Если истории меньше min_weeks, доверия к сезонности и тренду нет —
    возвращается EMA-оценка (avg_daily). Иначе считается линейно-сезонная
    оценка, и результат смешивается с EMA через вес trust_weight:
    чем длиннее история и горизонт, тем больше вес линейной модели.

    Args:
        weeks: недельный расход в базовых единицах.
        horizon_days: горизонт прогноза в днях.
        avg_daily: среднедневной расход (EMA), используется как база и для смеси.
        min_weeks: минимальная длина истории для подключения линейной модели.

    Returns:
        Среднедневной расход (смесь EMA и линейно-сезонной оценки).
    """

    ema_val = avg_daily

    if len(weeks) < min_weeks:
        return ema_val

    linear_val = _linear_seasonal_estimate(weeks, horizon_days, min_weeks)
    w = _trust_weight(len(weeks), horizon_days, min_weeks)
    return w * linear_val + (1 - w) * ema_val


def _trust_weight(n_weeks: int, horizon_days, min_weeks: int = config.MIN_WEEKS) -> float:
    """Считает вес линейно-сезонной модели в смеси с EMA.

    Вес растёт от 0 (только EMA) до 1 (только линейная модель) по мере
    увеличения длины истории. До min_weeks недель истории вес = 0.
    Дополнительно вес масштабируется горизонтом: чем длиннее горизонт,
    тем больше доверия к тренду и сезонности. Сверху вес ограничен 0.8,
    чтобы даже при очень длинной истории EMA оставалась в смеси.

    Args:
        n_weeks: длина недельной истории.
        horizon_days: горизонт прогноза в днях.
        min_weeks: минимальная длина истории для подключения линейной модели.

    Returns:
        Вес линейной модели в диапазоне [0.0, 0.8].
    """

    if n_weeks < min_weeks:
        return 0.0
    base = min(1.0, (n_weeks / 52 - 1) / 3)
    return min(0.8, base * (1 + horizon_days / 180))


def safety_stock_estimation(avg_daily_consumption: float, avg_daily_days: float) -> float:
    """Считает страховой запас как расход за заданное число дней.

    Args:
        avg_daily_consumption: среднедневной расход в базовых единицах.
        avg_daily_days: число дней страхового запаса (из справочника, safety_stock_days).

    Returns:
        Страховой запас в базовых единицах.
    """
    return avg_daily_consumption * avg_daily_days


def reorder_point_estimation(safety_stock: float, avg_daily: float, lead_time: float) -> float:
    """Считает точку заказа — уровень запаса, при котором пора оформлять закупку.

    Args:
        safety_stock: страховой запас в базовых единицах.
        avg_daily: среднедневной расход в базовых единицах.
        lead_time: срок поставки в днях (из справочника, lead_time_days).

    Returns:
        Точка заказа в базовых единицах.
    """
    return safety_stock + (avg_daily * lead_time)


def recommended_qty_estimation(current_stock: float, incoming_qty: float, reorder_point: float, forecast_demand: float,
                               pack_size: float) -> int:
    """Считает рекомендуемый объём закупки с округлением до упаковки.

    Логика:
        1. Доступный запас = остаток на складе + товар в пути.
        2. Позиция запаса = доступный запас − прогноз расхода на горизонт.
        3. Если позиция запаса выше точки заказа — заказ не нужен (0.0).
        4. Иначе заказываем столько, чтобы дотянуть позицию запаса до точки заказа.
        5. Итог округляется вверх до кратности упаковки (pack_size).

    Args:
        current_stock: текущий остаток на складе.
        incoming_qty: количество товара в пути.
        reorder_point: точка заказа в базовых единицах.
        forecast_demand: прогноз расхода на горизонт в базовых единицах.
        pack_size: кратность упаковки (сколько единиц в упаковке).

    Returns:
        Рекомендуемое количество к заказу, округлённое вверх до pack_size.
    """
    available = current_stock + incoming_qty

    if (available - forecast_demand) > reorder_point:
        result = 0.0
    else:
        result = reorder_point - (available - forecast_demand)

    return int(np.ceil(result / pack_size) * pack_size)


def estimated_cost_estimation(recommended_qty: float, price: float) -> float:
    """Считает стоимость рекомендуемого заказа.

    Args:
        recommended_qty: рекомендуемый объём закупки в базовых единицах.
        price: цена за базовую единицу из справочника.

    Returns:
        Стоимость заказа в рублях.
    """
    return recommended_qty * price


def stockout_date_estimation(current_stock: float, incoming_qty: float, avg_daily: float, as_of: str) -> str:
    """Возвращает дату исчерпания запаса (остаток + в пути) при текущем расходе.

    Если расхода нет или запас неположительный — возвращает as_of.

    Args:
        current_stock: остаток на складе.
        incoming_qty: товар в пути.
        avg_daily: среднедневной расход.
        as_of: дата остатков (ISO).

    Returns:
        Дата исчерпания в формате YYYY-MM-DD.
    """

    if avg_daily == 0.0:
        return as_of

    time = pd.to_datetime(as_of)

    total = current_stock + incoming_qty
    if total <= 0:
        return as_of

    days_left = total / avg_daily
    result = time + pd.Timedelta(days=days_left)

    return result.date().isoformat()


def confidence_estimation(weeks: list, horizon_days: int, weights: dict, alpha: float = 0.5) -> tuple:
    """Считает уверенность прогноза как взвешенную сумму пяти компонент.

    Компоненты:
        history    — длина истории (чем больше недель, тем выше).
        volatility — стабильность расхода (высокий разброс снижает).
        backtest   — точность EMA на исторических данных.
        agreement  — согласие EMA и линейно-сезонной модели.
        horizon    — чем длиннее горизонт, тем ниже уверенность.

    Args:
        weeks: недельный расход.
        horizon_days: горизонт прогноза в днях.
        weights: веса компонент.
        alpha: вес последней недели для EMA.

    Returns:
        Итоговый score (0..1) и словарь значений компонент.
    """
    components = {
        "history": _confidence_history(weeks),
        "volatility": _confidence_volatility(weeks),
        "backtest": _confidence_backtest(weeks, alpha),
        "agreement": _confidence_agreement(weeks, alpha),
        "horizon": _confidence_horizon(horizon_days)
    }
    score = sum(weights[k] * components[k] for k in weights)

    return float(score), components


def _confidence_horizon(horizon_days: int) -> float:
    """Компонента уверенности: горизонт прогноза.

    Чем длиннее горизонт, тем ниже уверенность.
    0 дней → 1.0, 90 дней → 0.5, дальше стремимся к 0.

    Args:
        horizon_days: горизонт прогноза в днях.

    Returns:
        Значение компоненты в диапазоне (0..1].
    """
    scale_days = config.SCALE_DAYS
    return 1.0 / (1.0 + horizon_days / scale_days)


def _confidence_history(weeks: list) -> float:
    """Компонента уверенности: длина истории.

    0 недель → 0.0, 52 недели → 1.0, дальше не растёт.

    Args:
        weeks: недельный расход.

    Returns:
        Значение компоненты в диапазоне [0.0, 1.0].
    """
    n = len(weeks)
    if n == 0:
        return 0.0
    return min(1.0, n / 52.0)


def _confidence_volatility(weeks: list) -> float:
    """Компонента уверенности: стабильность расхода.

    Считает коэффициент вариации (std / mean). Чем выше разброс
    относительно среднего, тем ниже уверенность. При mean <= 0
    берётся cv = 1.0. Пустая история → 0.0.

    Args:
        weeks: недельный расход.

    Returns:
        Значение компоненты в диапазоне (0.0, 1.0].
    """

    arr = np.asarray(weeks, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())

    cv = std / mean if mean > 0 else 1.0
    return 1.0 / (1.0 + cv)


def _confidence_backtest(weeks: list, weights_ema: float = 0.5) -> float:
    """Компонента уверенности: точность EMA на исторических данных.

    Идёт по неделям, предсказывает каждое значение предыдущим EMA
    и считает MAPE по ненулевым фактам. Чем меньше ошибка, тем выше
    уверенность. Если ошибок нет — 0.0.

    Args:
        weeks: недельный расход.
        weights_ema: вес последней недели для EMA (0..1).

    Returns:
        Значение компоненты в диапазоне [0.0, 1.0].
    """
    errors: list[float] = []
    ema = float(weeks[0])

    for actual in weeks[1:]:
        actual = float(actual)
        if actual != 0:
            errors.append(abs(ema - actual) / abs(actual))
        ema = weights_ema * actual + (1.0 - weights_ema) * ema

    if not errors:
        return 0.0

    mape = float(np.mean(errors))
    return max(0.0, 1.0 - mape)


def _confidence_agreement(weeks: list, weights_ema: float = 0.5) -> float:
    """Компонента уверенности: согласие EMA и линейно-сезонной модели.

    Сравнивает два прогноза на эталонном горизонте. Чем ближе значения,
    тем выше уверенность. Если оба прогноза неположительные — 0.0.

    Args:
        weeks: недельный расход.
        weights_ema: вес последней недели для EMA.

    Returns:
        Значение компоненты в диапазоне [0.0, 1.0].
    """
    reference_horizon = 7

    ema_val = avg_daily_consumption_estimation(weeks, weights_ema)
    linear_val = _linear_seasonal_est(weeks, reference_horizon)

    if ema_val + linear_val <= 0:
        return 0.0

    denom = (ema_val + linear_val) / 2.0
    if denom <= 0:
        return 0.0

    delta = abs(ema_val - linear_val) / denom
    return max(0.0, 1.0 - delta)


def explanation_estimation(result: dict, sku: str, horizon_days: int, item: str, components: dict, threshold):
    """Собирает текстовое объяснение расчёта для пользователя.

    Формирует человекочитаемый блок: средний дневной расход, прогноз на горизонт,
    остаток и товар в пути, страховой запас, точка заказа, рекомендованный объём
    и стоимость, дата исчерпания, уверенность прогноза и её компоненты.

    Если confidence ниже threshold, статус помечается как «требуется уточнение».

    Args:
        result: результат forecast_demand.
        sku: код позиции.
        horizon_days: горизонт прогноза в днях.
        item: запись из справочника (name, unit и т.д.).
        components: значения компонент уверенности.
        threshold: порог, ниже которого прогноз требует уточнения.

    Returns:
        Текст объяснения.
    """
    confidence = result["confidence"]

    status = "ТРЕБУЕТСЯ УТОЧНЕНИЕ" if confidence < threshold else "Достоверно"

    comps = ", ".join(f"{name}={value:.2f}" for name, value in components.items())

    available = result["current_stock"] + result["incoming_qty"]

    result = (
        f"{sku} ({item['name']}): \n"
        f"Средний дневной расход {sku} составляет {result['avg_daily_consumption']:.2f} {item['unit']}/день. \n"
        f"Предполагаемый расход {sku} с учетом сезонности на {horizon_days} дн составляет {result['forecast_demand']:.2f} {item['unit']}. \n"
        f"Остаток {result['current_stock']:.2f} {item['unit']} + "
        f"в пути {result['incoming_qty']:.2f} {item['unit']} "
        f"= доступно {available:.2f} {item['unit']}.\n"
        f"Страховой запас составляет {result['safety_stock']:.2f}, "
        f"точка заказа {result['reorder_point']:.2f}. \n"
        f"Рекомендуется закупить {result['recommended_qty']:.2f} {item['unit']} "
        f"на сумму {result['estimated_cost']:.2f} руб. \n"
        f"Ожидаемое исчерпание: {result['stockout_date']}. \n"
        f"Уверенность прогноза: {confidence:.2f} ({status}). \n"
        f"Компоненты: {comps}.\n"
    )
    return result








