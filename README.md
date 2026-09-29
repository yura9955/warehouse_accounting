# ИИ-помощник по складскому учёту и планированию закупок

Демонстрационный MVP: нормализация движений товара, прогноз потребности,
рекомендация закупки и разбор вопросов пользователя на естественном языке.

## Требования

- Python 3.11+
- Ключ DeepSeek API (опционально — без него работает fallback на правилах)

## Установка

pip install -r requirements.txt

## Настройка (опционально)

Создайте в корне файл `.env`:

    DEEPSEEK_API_KEY=sk-...

Без ключа `answer_question` использует rule-based классификатор
и извлечение сущностей регулярками.

## Запуск пайплайна

python main.py

Скрипт прогоняет нормализацию, прогноз и разбор 15 тестовых вопросов,
выводит результат в консоль.

## Тесты

    pytest -v

Если команда `pytest` не найдена, используйте:

    python -m pytest -v

## Архитектура

```
├── main.py                          точка входа, прогон пайплайна
│
├── data/
│   ├── catalog.json                 справочник товаров: unit, pack_size, min_order_qty,
│   │                                safety_stock_days, lead_time_days, price
│   ├── dictionaries.json            варианты написания месяцев, операций, единиц, SKU, локаций
│   ├── history.json                 недельный расход, остатки, поставки в пути (Часть 2)
│   └── Products_test.json           тексты M1–M8 (Часть 1)
│
├── normalization/                   Часть 1. Нормализация движений
│   ├── regex_parser.py              извлечение «сырых» фрагментов регулярками
│   ├── deterministic_parser.py      приведение к каноническому виду (даты, единицы, операции)
│   └── normalizer.py                Catalog, DataDictionaries, normalize_movement
│
├── forecasts/                       Часть 2. Прогноз и рекомендация закупки
│   ├── forecast.py                  EMA + линейная сезонность, confidence, recommended_qty
│   └── config.py                    WEIGHTS, PERIOD, MIN_WEEKS, SCALE_DAYS
│
├── aswers_to_questions/             Часть 3. Разбор вопросов
│   ├── answer_question.py           главная функция: разбор + forecast + текст ответа
│   ├── entity_extractor.py          SKU, локация, период, бюджет, load_growth
│   ├── intent_classifier.py         rule-based определение интента
│   ├── llm_client.py                DeepSeek через langchain-openai + fallback
│   └── config.py                    INTENT_KEYWORDS, UNKNOWN_PHRASES, пороги
│
└── tests/                           pytest
```

### `normalization/` — Часть 1

`regex_parser` извлекает «сырые» фрагменты регулярками, `find_sku_by_name`
сопоставляет позицию по стемам названия, `deterministic_parser` приводит
даты, операции, единицы и количества к каноническому виду. Публичная
точка входа — `normalize_movement(text)`, она сама подгружает `catalog.json`
и `dictionaries.json` из `data/`.

### `forecasts/` — Часть 2

`forecast_demand(history, sku, horizon_days, catalog, params)` считает
среднедневной расход через EMA, прогноз на горизонт, страховой запас,
точку заказа, рекомендованный объём с округлением до упаковки, стоимость
и дату исчерпания. Уверенность — взвешенная сумма компонент `history`,
`volatility`, `backtest`, `agreement`, `horizon`. Все веса и пороги
вынесены в `forecasts/config.py`.

### `aswers_to_questions/` — Часть 3

`answer_question(question, context)` проверяет стоп-фразы, разбирает
вопрос через DeepSeek (или правила при отсутствии ключа), извлекает SKU,
локацию, период, бюджет, вызывает `forecast_demand` и формирует текст
ответа. Числа берутся только из `forecast_demand` — LLM разбирает вопрос,
но не считает.