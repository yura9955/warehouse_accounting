from normalization import normalizer as nm
from forecasts import forecast as fc
from aswers_to_questions.answer_question import answer_question as aq
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

df_product = pd.read_json("data/Products_test.json")
catalog = nm.Catalog.load("data/catalog.json")

with open("data/history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

print("ЧАСТЬ 1")
for row_text in df_product["text"]:
    result = nm.normalize_movement(text = row_text)
    print(f"Для текста \"{row_text}\" результат:\n {result}")

print("ЧАСТЬ 2")

sku_full = ["OIL-001", "SCRB-020", "WRAP-030"]
list_horizon_days = [30, 90]
params = {"method": "auto", "weights_ema": 0.5, "threshold": 0.65}
for sku in sku_full:
    for horizon_days in list_horizon_days:
        result = fc.forecast_demand(history, sku = sku, horizon_days = horizon_days, catalog = catalog, params = params)
        print(f"Предсказание для \"{sku}\" на {horizon_days} дней:\n {result["explanation"]}")

print("ЧАСТЬ 3")

list_question = [
    "Сколько масла закупить на три месяца и сколько это будет стоить?",
    "Что нужно заказать в ближайшие 14 дней?",
    "Какой бюджет закупок на квартал?",
    "Что закончится до следующей поставки в Сочи?",
    "Какие партии сгорят в этом месяце?",
    "Насколько подорожало ароматическое масло у поставщика?",
    "Сколько альгинатной маски осталось в Красной Поляне?",
    "Посчитай закупку скраба на полгода при лимите 200 тысяч",
    "Почему план закупок вырос по сравнению с прошлым кварталом?",
    "Сколько стоит закупить всё, что в риске дефицита?",
    "Сколько масла уйдёт за месяц, если загрузка вырастет на 20%?",
    "Заказать масло",
    "Сколько это будет стоить?",
    "Когда привезут заказ от поставщика?",
    "Какая погода в Сочи на выходных?"
]
for question in list_question:
    parsed, conf, text = aq(question = question, context={"history": history, "catalog": catalog})
    print(f"Вопрос: \"{question}\"")
    print(f"Параметры запроса: {parsed}")
    print(f"Уровень уверенности: {conf}")
    print(f"Ответ: \n{text}\n")