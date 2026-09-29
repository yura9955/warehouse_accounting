import json
import os

try:
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_OK = True
except ImportError:  # пакет не установлен — работаем без LLM
    _LANGCHAIN_OK = False


_SYSTEM_PROMPT = (
    "Ты — ассистент по складскому учёту спа-оператора. "
    "Разбери вопрос пользователя и верни СТРОГО JSON без пояснений:\n"
    '{"intent": "<...>", "sku": <строка или null>, "location": <строка или null>, '
    '"period_days": <число или null>, "budget_limit": <число или null>, '
    '"confidence": <число от 0 до 1>}\n'
    "Возможные intent: forecast_purchase, reorder_list, budget, deficit_risk, "
    "expiry_risk, price_dynamics, unknown.\n"
    "НИЧЕГО не считай и не выдумывай числовые значения — только разбор."
)


def is_llm_available() -> bool:
    """True, если LLM можно вызвать (есть пакет и ключ)."""
    return _LANGCHAIN_OK and bool(os.getenv("DEEPSEEK_API_KEY"))


def ask_llm(question: str) -> dict | None:
    """Возвращает разобранный JSON от LLM или None при любой ошибке."""
    if not is_llm_available():
        return None
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
        )
        response = llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ])
        return _parse_json(getattr(response, "content", ""))
    except Exception:
        return None


def _parse_json(text: str) -> dict | None:
    """Достаёт JSON из ответа модели, убирая markdown-обёртку ```json ... ```."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None