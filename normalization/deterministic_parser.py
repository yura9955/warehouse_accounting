"""Приведение «сырых» фрагментов к каноническому виду.
Даты → ISO,
операции/локации/SKU → значения из dictionaries.json,
количества → число в базовой единице позиции из catalog.json.
"""

import re

def Deterministic_parser(result: dict, dicts_config: DataDictionaries, catalog: str) -> dict:
    if result["date"] is not None:
        result["date"] = parser_date(result["date"], dicts_config.months)

    if result["operation"] is not None:
        result["operation"] = parser_operation(result["operation"], dicts_config.operations)

    if result["sku"] is not None:
        result["sku"] = parser_sku(result["sku"], dicts_config.sku)

    if result["location"] is not None:
        result["location"] = parser_location(result["location"], dicts_config.location)

    if result["sku"] is not None:
        result["unit"] = parser_unit(result["sku"], catalog)

    if result["sku"] is not None:
        item = catalog.get(result["sku"])
        if item is not None:
            result["unit"] = item["unit"]

    result["qty"] = parser_qty(result["qty"], result["sku"], catalog)

    return result


def parser_unit(date: str, catalog: dict) -> str:
    return catalog.get(date)['unit']


def parser_date(date: str, dicts_config: dict) -> str:
    date_pars = re.split(r"[-./ ]", date)
    if len(date_pars[0]) == 4:  # Формат ГГГГ.ММ.ДД
        year, month, day = date_pars[0], date_pars[1], date_pars[2]
    else:  # Формат ДД.ММ.ГГГГ или ДД.ММ.ГГ
        day, month, year = date_pars[0], date_pars[1], date_pars[2]
        if len(year) == 2:
            year = "20" + year

    month = dicts_config.get(month, month).zfill(2)
    day = day.zfill(2)

    return f"{year}-{month}-{day}"


def parser_qty(qty_raw: str, sku: str, catalog: Catalog) -> float:

    UNIT_CONVERSIONS: dict[tuple[str, str], float] = {("мл", "л"): 0.001,
                                                      ("л", "мл"): 1000.0,
                                                      ("г", "кг"): 0.001,
                                                      ("кг", "г"): 1000.0}
    if not qty_raw:
        return None

    item = catalog.get(sku) if sku else None
    base_unit = item["unit"] if item else None

    text = qty_raw.lower().replace(",", ".").replace("\u2212", "-")

    m = re.search(r"(\d+(?:\.\d+)?)\D+по\s*(-?\d+(?:\.\d+)?)\s*([а-яa-z]+\.?)", text)
    if m:
        qty = float(m.group(1)) * float(m.group(2))
        unit_raw = m.group(3).rstrip(".")
    else:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([а-яa-z]+\.?)", text)
        if not m:
            return None
        qty = float(m.group(1))
        unit_raw = m.group(2).rstrip(".")

    if base_unit and unit_raw != base_unit:
        factor = UNIT_CONVERSIONS.get((unit_raw, base_unit))
        if factor is not None:
            qty *= factor
    return qty


def parser_operation(operation: str, dicts_config: dict) -> str:
    return dicts_config[operation]

def parser_sku(sku: str, dicts_config: dict) -> str:
    return dicts_config[sku]

def parser_location(location: str, dicts_config: dict) -> str:
    return dicts_config[location]