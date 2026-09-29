import re
import json
from pathlib import Path
from . import config


def extract_entities(question: str, catalog) -> dict:
    """Возвращает словарь с сущностями вопроса.

    Args:
        question: вопрос пользователя.
        catalog:  экземпляр normalizer.Catalog.

    Returns:
        {"sku": str | None, "location": str | None,
         "period_days": int | None, "budget_limit": float | None}.
    """
    text = question.lower()
    return {
        "sku": extract_sku(text, catalog),
        "location": extract_location(text),
        "period_days": extract_period(text),
        "budget_limit": extract_budget(text),
    }


def extract_sku(text: str, catalog) -> str | None:
    """Ищет SKU по явному коду (OIL-001) или по названию («масло» → OIL-001)."""
    match = re.search(r"\b([a-z]{2,5})[-_ ]?(\d{2,4})\b", text)
    if match:
        candidate = f"{match.group(1).upper()}-{match.group(2)}"
        if catalog.get(candidate):
            return candidate

    return _find_sku_by_name(text, catalog)


def _find_sku_by_name(text: str, catalog) -> str | None:
    """Сопоставляет слова вопроса с названиями по 4-буквенным стемам.

    Например, «масла» и «масло» → общий стем «масл». Возвращает SKU
    с максимальным числом общих стемов или None.
    """
    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    q_stems = {
        w[:config.SKU_STEM_LEN]
        for w in tokens
        if len(w) >= config.SKU_STEM_LEN
    }
    if not q_stems:
        return None

    counts: dict[str, int] = {}
    for sku, item in catalog.by_sku.items():
        name_stems = {
            w[:config.SKU_STEM_LEN]
            for w in re.findall(r"[а-яёa-z]+", item["name"].lower())
            if len(w) >= config.SKU_STEM_LEN
        }
        common = len(q_stems & name_stems)
        if common:
            counts[sku] = common

    if not counts:
        return None
    return max(counts, key=counts.get)


def extract_location(text: str) -> str | None:
    """Ищет локацию в тексте по стемам вариантов из dictionaries.json.

    Возвращает каноническое имя (как в справочнике) или None.
    Стем-матчинг позволяет ловить падежные формы: «в Красной Поляне»
    сматчится с вариантом «красная поляна» (стемы «красн» + «полян»).
    """
    dicts_path = Path(__file__).resolve().parent.parent / "data" / "dictionaries.json"
    stem_len = 4

    with open(dicts_path, encoding="utf-8") as f:
        variants_by_location = json.load(f).get("location", {})

    low = text.lower()
    q_stems = {w[:stem_len] for w in re.findall(r"[а-яёa-z0-9]+", low)}
    if not q_stems:
        return None

    for canonical, variants in variants_by_location.items():
        for variant in variants:
            v_stems = {
                w[:stem_len]
                for w in re.findall(r"[а-яёa-z0-9]+", variant.lower())
            }
            if v_stems and v_stems <= q_stems:
                return canonical
    return None


def extract_period(text: str) -> int | None:
    """Преобразует словесный или числовой период в количество дней."""
    # «три месяца», «два месяца»
    pattern = r"\b(" + "|".join(config.PERIOD_NUM_WORDS) + r")\s+месяц"
    match = re.search(pattern, text)
    if match:
        return config.PERIOD_NUM_WORDS[match.group(1)] * 30

    if "полгода" in text or "пол года" in text:
        return 180
    if "квартал" in text:
        return 90
    if re.search(r"\bгод\b", text) or re.search(r"\bлет\b", text):
        return 365

    # «14 дней», «2 недели», «3 месяца»
    match = re.search(
        r"(\d+)\s*(дн\w*|день|дня|дней|недел\w*|месяц\w*|год\w*|лет)",
        text,
    )
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("д"):
            return n
        if unit.startswith("недел"):
            return n * 7
        if unit.startswith("месяц"):
            return n * 30
        if unit.startswith(("год", "лет")):
            return n * 365

    # «в этом месяце», «на месяц»
    if re.search(r"\bмесяц", text):
        return 30
    return None


def extract_budget(text: str) -> float | None:
    """Извлекает бюджетный лимит («200 тысяч», «лимит 500000»)."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(тыс\w*|млн\w*|руб\w*|₽)", text)
    if match:
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if unit.startswith("тыс"):
            return value * 1_000
        if unit.startswith("млн"):
            return value * 1_000_000
        return value

    match = re.search(r"лимит[а-я]*\s+(\d+(?:[.,]\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None

def extract_load_growth(text: str) -> float | None:
    """Извлекает коэффициент роста загрузки из вопроса («+20%» → 0.2)."""
    _LOAD_GROWTH_RE = re.compile(r"загрузк\w*\s+(?:выраст\w*|увелич\w*)\s+на\s+(\d+)\s*%")
    m = _LOAD_GROWTH_RE.search(text.lower())
    return int(m.group(1)) / 100.0 if m else None