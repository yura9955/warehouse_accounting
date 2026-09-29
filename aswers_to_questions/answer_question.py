from datetime import date

from forecasts import forecast as fc
from . import config
from .entity_extractor import extract_entities, extract_load_growth
from .intent_classifier import classify_intent
from .llm_client import ask_llm, is_llm_available


# Обязательные сущности по интентам. Если хотя бы одна отсутствует —
# вопрос переводится в unknown с уточняющим вопросом.
_REQUIRED_ENTITIES: dict[str, list[str]] = {
    "forecast_purchase": ["sku", "period_days"],
    "reorder_list": [],
    "budget": [],
    "deficit_risk": [],
    "expiry_risk": [],
    "price_dynamics": [],
    "unknown": [],
}

_CLARIFY_ASKS: dict[str, str] = {
    "sku": "какой именно товар (SKU) вас интересует",
    "period_days": "на какой период (дни, месяцы, кварталы) нужен расчёт",
    "budget_limit": "какой бюджетный лимит использовать",
}


def answer_question(question: str, context: dict) -> tuple[dict, float, str]:
    """Разбирает вопрос, вызывает расчётный модуль и формирует текстовый ответ.

    Args:
        question: вопрос пользователя на естественном языке.
        context: {
            "history": dict — данные Части 2 (as_of, weekly_consumption,
                       current_stock, incoming_qty);
            "catalog": normalizer.Catalog — справочник товаров;
            "params": dict — параметры forecast_demand (опционально).
        }

    Returns:
        Кортеж (parsed, confidence, answer):
            parsed     — {"intent": ..., "sku": ..., "location": ...,
                          "period_days": ..., "budget_limit": ...};
            confidence — уверенность разбора, 0.0..1.0;
            answer     — текст ответа (числа только из forecast_demand).
    """
    if not context or "catalog" not in context:
        empty = {"intent": "unknown", "sku": None, "location": None,
                 "period_days": None, "budget_limit": None}
        return empty, 0.0, "Не передан справочник товаров (context['catalog'])."
    catalog = context["catalog"]
    history = context.get("history") or {}
    params = {**config.DEFAULT_FORECAST_PARAMS, **context.get("params", {})}

    text_low = question.lower()
    if any(phrase in text_low for phrase in config.UNKNOWN_PHRASES):
        empty = {"intent": "unknown", "sku": None, "location": None,
                 "period_days": None, "budget_limit": None}
        return empty, 0.0, (
            "Этот вопрос вне возможностей MVP. Уточните параметры закупки: "
            "товар, период, локацию или бюджет."
        )

    intent, intent_conf, entities = _parse_question(question, catalog)
    entities["load_growth"] = extract_load_growth(question)
    required = _REQUIRED_ENTITIES.get(intent, [])
    missing = [name for name in required if entities.get(name) is None]
    completeness = _completeness(required, missing)

    confidence = round(intent_conf * completeness, 2)

    if intent == "unknown" or intent_conf < config.INTENT_GAP_THRESHOLD or missing:
        parsed = {"intent": "unknown", **entities}
        return parsed, 0.0, _clarifying_question(intent, missing)

    parsed = {"intent": intent, **entities}
    answer = _build_answer(intent, entities, history, catalog, params)
    return parsed, confidence, answer


def _parse_question(question: str, catalog) -> tuple[str, float, dict]:
    """Разбирает вопрос: сначала LLM, при неудаче или unknown — правила."""
    if is_llm_available():
        llm_data = ask_llm(question)
        if llm_data:
            intent = llm_data.get("intent", "unknown")
            if intent not in config.INTENT_KEYWORDS and intent != "unknown":
                intent = "unknown"
            confidence = float(llm_data.get("confidence", 0.5))
            entities = {
                "sku": _validate_sku(llm_data.get("sku"), catalog),
                "location": llm_data.get("location"),
                "period_days": llm_data.get("period_days"),
                "budget_limit": llm_data.get("budget_limit"),
            }
            rule_entities = extract_entities(question, catalog)
            for key, value in rule_entities.items():
                if entities.get(key) is None and value is not None:
                    entities[key] = value

            # Если LLM не понял интент — пробуем правила.
            if intent == "unknown":
                rule_intent, rule_conf, _ = classify_intent(question)
                if rule_intent != "unknown" and rule_conf >= config.INTENT_GAP_THRESHOLD:
                    return rule_intent, rule_conf, entities

            return intent, confidence, entities

    intent, confidence, _ = classify_intent(question)
    entities = extract_entities(question, catalog)
    return intent, confidence, entities


def _validate_sku(sku, catalog) -> str | None:
    """Приводит SKU к каноническому виду и проверяет по справочнику."""
    if not sku or not isinstance(sku, str):
        return None
    normalized = sku.upper().replace("_", "-").replace(" ", "-")
    return normalized if catalog.get(normalized) else None


def _completeness(required: list[str], missing: list[str]) -> float:
    """Доля заполненных обязательных сущностей, 0.0..1.0."""
    if not required:
        return 1.0
    return (len(required) - len(missing)) / len(required)


def _clarifying_question(intent: str, missing: list[str]) -> str:
    """Формулирует уточняющий вопрос, если разбор неоднозначен или неполон."""
    if intent == "unknown":
        return (
            "Не удалось однозначно разобрать вопрос. Уточните, пожалуйста, "
            "что вас интересует: прогноз закупки товара, список позиций к заказу, "
            "бюджет, риск дефицита, риск списания или динамика цен."
        )
    parts = [_CLARIFY_ASKS[name] for name in missing if name in _CLARIFY_ASKS]
    if not parts:
        return "Требуется уточнение параметров запроса."
    return "Уточните, пожалуйста: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Формирование ответов по интентам
# ---------------------------------------------------------------------------


def _build_answer(intent: str, entities: dict, history: dict, catalog, params: dict) -> str:
    """Диспетчер: выбирает формирователь ответа по интенту."""
    if intent == "forecast_purchase":
        return _answer_forecast_purchase(entities, history, catalog, params)
    if intent == "reorder_list":
        return _answer_reorder_list(entities, history, catalog, params)
    if intent == "budget":
        return _answer_budget(entities, history, catalog, params)
    if intent == "deficit_risk":
        return _answer_deficit_risk(entities, history, catalog, params)
    if intent == "expiry_risk":
        return (
            "В MVP нет данных о сроках годности партий, поэтому оценить риск списания "
            "невозможно. Требуется подключение источника данных о партиях."
        )
    if intent == "price_dynamics":
        return (
            "В MVP нет исторических цен, поэтому динамику цен построить нельзя. "
            "Требуется подключение источника закупочных цен."
        )
    return "Не удалось обработать запрос."


def _answer_forecast_purchase(entities: dict, history: dict, catalog, params: dict) -> str:
    """Считает прогноз по одному SKU и собирает ответ с числами из forecast_demand."""
    sku = entities["sku"]
    horizon = entities["period_days"]
    item = catalog.get(sku)
    if item is None:
        return f"Позиция {sku} не найдена в справочнике."

    result = fc.forecast_demand(history, sku, horizon, catalog, params)
    if result.get("avg_daily_consumption") is None:
        return f"Недостаточно данных для расчёта {sku}: {result.get('explanation')}"

    lines = [
        f"Прогноз закупки {sku} ({item['name']}) на {horizon} дн.",
        f"Средний дневной расход: {result['avg_daily_consumption']:.2f} {item['unit']}.",
        f"Прогноз расхода: {result['forecast_demand']:.2f} {item['unit']}.",
        f"Остаток: {result['current_stock']:.2f} + в пути: {result['incoming_qty']:.2f} {item['unit']}.",
        f"Страховой запас: {result['safety_stock']:.2f}, точка заказа: {result['reorder_point']:.2f}.",
        f"Рекомендуется закупить: {result['recommended_qty']:.2f} {item['unit']}.",
        f"Стоимость: {result['estimated_cost']:.2f} руб.",
        f"Дата исчерпания: {result['stockout_date']}.",
        f"Уверенность прогноза: {result['confidence']:.2f}.",
    ]

    limit = entities.get("budget_limit")
    if limit is not None:
        if result["estimated_cost"] > limit:
            diff = result["estimated_cost"] - limit
            lines.append(f"Стоимость превышает лимит {limit:.0f} руб. на {diff:.2f} руб.")
        else:
            lines.append(f"Стоимость в пределах лимита {limit:.0f} руб.")
    growth = entities.get("load_growth")
    if growth is not None:
        lines.append(
            f"Оговорка: рост загрузки на {int(growth * 100)}% в MVP не учитывается — "
            f"модель строит прогноз по историческому расходу. "
            f"Для учёта загрузки нужен источник плановой загрузки филиалов."
        )
    return "\n".join(lines)


def _answer_reorder_list(entities: dict, history: dict, catalog, params: dict) -> str:
    """Список позиций, требующих заказа на заданном горизонте.

    Если пользователь указал локацию, честно сообщаем, что разбивки
    остатков по локациям в MVP нет, и показываем данные по всему складу.
    """
    horizon = entities.get("period_days") or config.DEFAULT_HORIZON_DAYS
    location = entities.get("location")

    lines = [f"Позиции к заказу на ближайшие {horizon} дн.:"]
    if location:
        lines.append(
            f"(локация «{location}»: разбивки остатков по локациям в MVP нет, "
            f"показаны данные по всему складу)"
        )

    found = False
    for sku in history.get("weekly_consumption", {}):
        item = catalog.get(sku)
        if item is None:
            continue
        result = fc.forecast_demand(history, sku, horizon, catalog, params)
        if result.get("recommended_qty"):
            found = True
            lines.append(
                f"- {sku} ({item['name']}): {result['recommended_qty']:.0f} {item['unit']}, "
                f"{result['estimated_cost']:.2f} руб., "
                f"исчерпание {result['stockout_date']}"
            )

    if not found:
        lines.append("Позиций, требующих заказа, не обнаружено.")
    return "\n".join(lines)


def _answer_budget(entities: dict, history: dict, catalog, params: dict) -> str:
    """Суммарный бюджет закупок на период и сверка с лимитом."""
    horizon = entities.get("period_days") or config.DEFAULT_HORIZON_DAYS
    lines = [f"Бюджет закупок на {horizon} дн.:"]
    total = 0.0

    for sku in history.get("weekly_consumption", {}):
        item = catalog.get(sku)
        if item is None:
            continue
        result = fc.forecast_demand(history, sku, horizon, catalog, params)
        cost = result.get("estimated_cost") or 0.0
        if cost > 0:
            total += cost
            lines.append(
                f"- {sku}: {result['recommended_qty']:.0f} {item['unit']} = {cost:.2f} руб."
            )

    lines.append(f"Итого: {total:.2f} руб.")

    limit = entities.get("budget_limit")
    if limit is not None:
        if total > limit:
            lines.append(f"Превышение лимита {limit:.0f} руб. на {total - limit:.2f} руб.")
        else:
            lines.append(f"В пределах лимита {limit:.0f} руб. (запас {limit - total:.2f} руб.)")
    return "\n".join(lines)


def _answer_deficit_risk(entities: dict, history: dict, catalog, params: dict) -> str:
    """Позиции с риском дефицита. Если пользователь указал конкретный SKU,
    показываем только его (с остатком и оценкой, хватит ли до поставки)."""
    as_of = history.get("as_of")
    horizon = entities.get("period_days") or config.DEFAULT_HORIZON_DAYS
    target_sku = entities.get("sku")     # ← новое

    lines = ["Позиции с риском дефицита:"]
    found = False
    total_cost = 0.0

    skus = [target_sku] if target_sku else list(history.get("weekly_consumption", {}))
    for sku in skus:
        item = catalog.get(sku)
        if item is None:
            continue
        result = fc.forecast_demand(history, sku, horizon, catalog, params)
        stockout = result.get("stockout_date")
        if not stockout or not as_of:
            continue
        try:
            days_left = (date.fromisoformat(stockout) - date.fromisoformat(as_of)).days
        except ValueError:
            continue
        lead = item["lead_time_days"]

        if days_left <= lead:
            found = True
            cost = result.get("estimated_cost") or 0.0
            total_cost += cost
            lines.append(
                f"- {sku} ({item['name']}): хватит на ~{days_left} дн "
                f"(lead time {lead} дн), заказать {result['recommended_qty']:.0f} "
                f"{item['unit']} = {cost:.2f} руб."
            )
        elif target_sku:
            # Пользователь спросил про конкретный SKU — отвечаем даже если риска нет
            found = True
            lines.append(
                f"- {sku} ({item['name']}): остаток {result['current_stock']:.2f} "
                f"{item['unit']}, хватит на ~{days_left} дн (lead time {lead} дн). "
                f"Риска дефицита нет."
            )

    if not found:
        lines.append("Позиций в риске дефицита не обнаружено.")
    elif total_cost > 0:
        lines.append(f"Итого стоимость закупки: {total_cost:.2f} руб.")
    return "\n".join(lines)