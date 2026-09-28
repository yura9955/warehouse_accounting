import numpy as np
import pandas as pd

def forecast_demand(history: dict, sku: str, horizon_days: int, catalog: dict, params: dict | None = None) -> dict:
    """
    k - размер окна для скользящего среднего
    sliding_window_or_linear - параметр для выбора скользящего окна или линейной регрессии (False - Скользящее окно. True - Регрессия)
    params = {"alpha_avg_daily": 0.5,
              "alpha_recommended_qty": 0.5,
              "method": False}
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
    if history is None:
        return result

    result["current_stock"] = history["current_stock"][sku]

    result["incoming_qty"] = history["incoming_qty"][sku]

    result["avg_daily_consumption"] = avg_daily_consumption_estimation(history["weekly_consumption"][sku],
                                                                       alpha=0.5)

    result["forecast_demand"] = forecast_demand_estimation(result["avg_daily_consumption"],
                                                           horizon_days)

    result["safety_stock"] = safety_stock_estimation(result["avg_daily_consumption"],
                                                     catalog.get(sku)["safety_stock_days"])

    result["reorder_point"] = reorder_point_estimation(result["safety_stock"],
                                                       result["avg_daily_consumption"],
                                                       catalog.get(sku)["lead_time_days"])

    result["recommended_qty"] = recommended_qty_estimate(weeks = history["weekly_consumption"][sku],
                                                         horizon_days = horizon_days)

    result["estimated_cost"] = estimated_cost_estimate(recommended_qty = result["recommended_qty"],
                                                       price = catalog.get(sku)["price"])

    result["stockout_date"] = stockout_date_estimate(current_stock = result["current_stock"],
                                                     incoming_qty = result["incoming_qty"],
                                                     avg_daily = result["avg_daily_consumption"],
                                                     as_of = history["as_of"])

    result["confidence"] = confidence_estimate(weeks = history["weekly_consumption"][sku],
                                               ema_val = ema_estimate(history["weekly_consumption"][sku]),
                                               linear_val = linear_seasonal_estimate(history["weekly_consumption"][sku], horizon_days))

    result["explanation"] = None

    return result


def explanation_estimate():
    pass


def confidence_estimate(weeks: list[float], ema_val: float | None = None, linear_val: float | None = None) -> float:
    c_history = min(1.0, np.log(1 + len(weeks)) / np.log(52))

    mean_data = np.mean(weeks)
    std_data = np.std(weeks)
    cv = std_data / mean_data
    c_volatility = max(0.0, 1 - float(cv) / 2)

    c_agreement = 1.0
    """
    if ema_val is not None and linear_val is not None:
        if ema_val > 0 and linear_val > 0:
        ratio = min(ema_val, linear_val) / max(ema_val, linear_val)
        c_agreement = max(0.0, 2 * ratio - 1)
    print(c_history, c_volatility, c_agreement)
    """
    return float(c_history * c_volatility)






def stockout_date_estimate(current_stock: float, incoming_qty: float, avg_daily: float, as_of: str) -> str:

    time = pd.to_datetime(as_of)

    total = current_stock + incoming_qty
    if total <= 0:
        return as_of

    days_left = total / avg_daily
    result = time+pd.Timedelta(days=days_left)

    return result.date().isoformat()


def estimated_cost_estimate(recommended_qty: float, price: float) -> float:
    return recommended_qty * price

def recommended_qty_estimate(weeks: list[float], horizon_days, method: str = "auto", alpha: float = 0.5) -> float:
    """ Функция для расчета количества закупки материалов """
    if not weeks:
        return 0.0

    if method == "ema":
        result = ema_estimate(weeks, alpha)
    elif method == "linear":
        result = linear_seasonal_estimate(weeks, horizon_days)
    elif method == "auto":
        result = auto_estimate(weeks, horizon_days, alpha)
    else:
        raise ValueError(f"unknown method: {method}")

    return result * horizon_days


def ema_estimate(weeks: list[float], alpha: float = 0.5) -> float:
    """Экспоненциальное сглаживание. alpha — вес последней недели (0..1)."""
    result = weeks[0]
    for w in weeks[1:]:
        result = alpha * w + (1 - alpha) * result
    return result / 7


def linear_seasonal_estimate(weeks: list[float], horizon_days, period=12) -> float:
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


def auto_estimate(weeks: list[float], horizon_days, alpha: float, min_weeks: int = 52) -> float:
    """EMA до года, дальше — смесь с Linear. После 4 лет только Linear"""
    ema_val = ema_estimate(weeks, alpha)

    if len(weeks) < min_weeks:
        return ema_val

    linear_val = linear_seasonal_estimate(weeks, horizon_days, min_weeks)
    w = trust_weight(len(weeks), horizon_days, min_weeks)
    return w * linear_val + (1 - w) * ema_val


def trust_weight(n_weeks: int, horizon_days, min_weeks: int = 52) -> float:
    """
    Вес Linear. 0 = только EMA, 1 = только Linear.
    К 4 годам достигает 1.0.
    """
    if n_weeks < min_weeks:
        return 0.0
    base = min(1.0, (n_weeks / 52 - 1) / 3)
    return min(1.0, base * (1 + horizon_days / 180))


def reorder_point_estimation(safety_stock: float, avg_daily: float, lead_time: float) -> float:
    return safety_stock + (avg_daily * lead_time)


def safety_stock_estimation(avg_daily_consumption: float, avg_daily_days: float) -> float:
    return avg_daily_consumption * avg_daily_days


def avg_daily_consumption_estimation(weeks: list[float], alpha: float = 0.5) -> float:
    """
    Функция для расчета дневного расхода материала по методу экспоненциальное скользящее среднее
    alpha принимает значениея от 0 до 1.
    Чем меньше alpha тем больше учитываються старые недели.
    """
    if not weeks:
        return 0.0
    result = weeks[0]

    for week in weeks[1:]:
        result = alpha * week + (1 - alpha) * result

    return result / 7.0


def forecast_demand_estimation(avg_daily: float, horizon_days: int) -> float:
    """ Функция для расчета расход за период, на который строим прогноз """
    return avg_daily * horizon_days








