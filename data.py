import os
import pandas as pd
import yaml

# ---------------------------------
# Load config
# ---------------------------------
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

RAW_PATH = config["data"]["raw_path"]
STORE_ID = config["data"]["store"]

HORIZONS = config["model"]["horizons"]
INPUT_SIZE = config["model"]["input_size"]
MIN_REQUIRED_LENGTH = INPUT_SIZE + max(HORIZONS)

# ---------------------------------
# Output folder
# ---------------------------------
output_path = "data_processed"
os.makedirs(output_path, exist_ok=True)

# ---------------------------------
# 1️ Load raw data
# ---------------------------------
calendar = pd.read_csv(os.path.join(RAW_PATH, "calendar.csv"))
sales = pd.read_csv(os.path.join(RAW_PATH, "sales_train_evaluation.csv"))
sell_prices = pd.read_csv(os.path.join(RAW_PATH, "sell_prices.csv"))

# ---------------------------------
# 2️ Filter store
# ---------------------------------
sales_ca1 = sales[sales["store_id"] == STORE_ID].copy()
sell_prices_ca1 = sell_prices[sell_prices["store_id"] == STORE_ID].copy()

# Drop unused columns
sales_ca1 = sales_ca1.drop(columns=["id", "dept_id", "cat_id", "state_id", "store_id"])
sell_prices_ca1 = sell_prices_ca1.drop(columns=["store_id"])

# ---------------------------------
# 3️ Filter items with sufficient history
# ---------------------------------
calendar["date"] = pd.to_datetime(calendar["date"])
calendar = calendar.sort_values("date")

wm_to_last_day = calendar.groupby("wm_yr_wk")["date"].max().to_dict()

item_weeks = sell_prices_ca1.groupby("item_id")["wm_yr_wk"].agg(["min","max"]).reset_index()
item_weeks = item_weeks[item_weeks["max"] >= calendar["wm_yr_wk"].max()].copy()
item_weeks["first_sold_date"] = item_weeks["min"].map(wm_to_last_day)
last_available_date = calendar["date"].max()
item_weeks["days_available"] = (last_available_date - item_weeks["first_sold_date"]).dt.days

# Filter by MIN_REQUIRED_LENGTH
total_products = item_weeks["item_id"].nunique()
valid_items = item_weeks[item_weeks["days_available"] >= MIN_REQUIRED_LENGTH]["item_id"].tolist()
valid_products = len(valid_items)
filtered_products = total_products - valid_products
filtered_pct = 100 * filtered_products / total_products

sales_ca1 = sales_ca1[sales_ca1["item_id"].isin(valid_items)].copy()

# ---------------------------------
# 4️ Convert sales to long format
# ---------------------------------
day_cols = [c for c in sales_ca1.columns if c.startswith("d_")]
sales_long = sales_ca1.melt(
    id_vars=["item_id"],
    value_vars=day_cols,
    var_name="d",
    value_name="y"
)

d_to_date = dict(zip(calendar["d"], calendar["date"]))
sales_long["ds"] = sales_long["d"].map(d_to_date)
sales_long = sales_long.rename(columns={"item_id":"unique_id"})
sales_long = sales_long[["unique_id","ds","y"]]
sales_long = sales_long.sort_values(["unique_id","ds"]).reset_index(drop=True)

# ---------------------------------
# 5️ Remove duplicates
# ---------------------------------
before = len(sales_long)
sales_long = sales_long.drop_duplicates(["unique_id","ds"])
after = len(sales_long)
print(f"Removed {before - after} duplicate rows")

# ---------------------------------
# 6️ Structural missing analysis
# ---------------------------------
global_min = sales_long["ds"].min()
global_max = sales_long["ds"].max()
actual_counts = sales_long.groupby("unique_id")["ds"].nunique()
n_days = (global_max - global_min).days + 1
expected_counts = pd.Series(n_days, index=actual_counts.index)
missing_per_product = expected_counts - actual_counts
total_missing = missing_per_product.sum()
total_expected = len(actual_counts) * n_days
missing_pct = 100 * total_missing / total_expected

# ---------------------------------
# 7️ Create full date grid per product
# ---------------------------------
start_dates_df = item_weeks[["item_id","first_sold_date"]].rename(columns={"item_id":"unique_id"})
calendar_dates = calendar[["date"]].rename(columns={"date":"ds"})
calendar_dates["key"] = 1
start_dates_df["key"] = 1

full_index = pd.merge(start_dates_df, calendar_dates, on="key").drop("key", axis=1)
full_index = full_index[full_index["ds"] >= full_index["first_sold_date"]].drop("first_sold_date", axis=1)

sales_long = pd.merge(full_index, sales_long, on=["unique_id","ds"], how="left")
sales_long["y"] = sales_long["y"].fillna(0).clip(lower=0)
sales_long = sales_long.sort_values(["unique_id","ds"]).reset_index(drop=True)

# ---------------------------------
# 8️ Merge exogenous features
# ---------------------------------
calendar_exo = calendar[[
    "date","weekday","event_name_1","event_name_2","snap_CA"
]].rename(columns={"date":"ds","snap_CA":"snap"})

sales_long = sales_long.merge(calendar_exo, on="ds", how="left")

# Weekday mapping
weekday_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,
    "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
}
sales_long["weekday"] = sales_long["weekday"].map(weekday_map)

# Time features
sales_long["day_of_week"] = sales_long["ds"].dt.weekday
sales_long["month"] = sales_long["ds"].dt.month

# Weekend flag
sales_long["is_weekend"] = sales_long["day_of_week"].isin([5,6]).astype(int)

# Event flag
sales_long["event_flag"] = (
    (sales_long["event_name_1"].notna()) |
    (sales_long["event_name_2"].notna())
).astype(int)

# Map ds → wm_yr_wk for price merge
date_to_wm = dict(zip(calendar["date"], calendar["wm_yr_wk"]))
sales_long["wm_yr_wk"] = sales_long["ds"].map(date_to_wm)

# Merge sell prices
sell_prices_ca1 = sell_prices_ca1.rename(columns={"item_id":"unique_id"})
sales_long = sales_long.merge(sell_prices_ca1, on=["unique_id","wm_yr_wk"], how="left")

# Fill missing prices
sales_long["sell_price"] = sales_long.groupby("unique_id")["sell_price"].ffill().fillna(0)

# Price dynamics
sales_long["price_change"] = (
    sales_long.groupby("unique_id")["sell_price"]
    .pct_change()
    .fillna(0)
)

# ---------------------------------
# 9 Create lag features
# ---------------------------------
sales_long = sales_long.sort_values(["unique_id", "ds"])

sales_long["lag_1"] = sales_long.groupby("unique_id")["y"].shift(1)
sales_long["lag_7"] = sales_long.groupby("unique_id")["y"].shift(7)

# Fill missing lags with 0 (important for sparse data)
sales_long["lag_1"] = sales_long["lag_1"].fillna(0)
sales_long["lag_7"] = sales_long["lag_7"].fillna(0)

sales_long["was_zero_yesterday"] = (sales_long["lag_1"] == 0).astype(int)

# ---------------------------------
# 10 Generate data report
# ---------------------------------
n_series = sales_long["unique_id"].nunique()
date_range = (sales_long["ds"].min(), sales_long["ds"].max())
top_series = sales_long.groupby("unique_id")["y"].sum().sort_values(ascending=False).head(10)

report = f"""
Processed M5 {STORE_ID} Dataset Report
--------------------------------
Total products before filtering: {total_products}
Number of time series (after filtering): {n_series}
Products removed due to insufficient history (<{MIN_REQUIRED_LENGTH} days): {filtered_products}
Percentage of products filtered out: {filtered_pct:.2f} %
Date range: {date_range[0]} → {date_range[1]}
Total expected rows: {len(sales_long)}
Total structurally missing timestamps: {total_missing}
Missing percentage (before filling): {missing_pct:.3f} %

Top 10 series by total sales volume:
{top_series}
"""

print(report)

with open(os.path.join(output_path,"data_report.txt"), "w") as f:
    f.write(report)

# ---------------------------------
# 11 Save final cleaned dataset
# ---------------------------------
model_columns = [
    "unique_id",
    "ds",
    "y",
    "is_weekend",
    "snap",
    "sell_price",
    "price_change",
    "event_flag",
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "was_zero_yesterday"
]

sales_model = sales_long[model_columns].copy()

final_path = os.path.join(output_path, "sales_ca1_clean.csv")
sales_model.to_csv(final_path, index=False)

print("Final cleaned dataset saved to:", final_path)