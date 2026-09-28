from normalization import normalizer as nm
from forecasts import forecast as fc
import pandas as pd
import json

df_product = pd.read_json("data/Products_test.json")
catalog = nm.Catalog.load("data/catalog.json")
data_dict = nm.DataDictionaries.load("data/dictionaries.json")

with open("data/Expenditure.json", "r", encoding="utf-8") as f:
    Expenditure = json.load(f)

#for row_text in df_product["text"]:
#    result = nm.normalize_movement(row_text, catalog, data_dict)
result = fc.forecast_demand(Expenditure, sku = "OIL-001", horizon_days = 30, catalog = catalog)
print(result)