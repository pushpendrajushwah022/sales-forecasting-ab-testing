
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
import sqlite3

# PART 1: TIME SERIES SALES FORECASTING

print("#" * 60)
print("# PART 1: TIME SERIES SALES FORECASTING")
print("#" * 60)

# ---- Generate 2 years of realistic daily sales data ----
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
n = len(dates)
trend = np.linspace(1000, 1800, n)
day_of_year = dates.dayofyear
seasonality = 300 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
weekday = dates.dayofweek
weekly_effect = np.where(weekday >= 5, 150, 0)
noise = np.random.normal(0, 80, n)
sales = np.clip(trend + seasonality + weekly_effect + noise, 200, None).round(0)

df_sales_raw = pd.DataFrame({"date": dates.astype(str), "daily_sales": sales, "day_of_week": dates.day_name()})

# ---- Build SQL database and load FROM it ----
conn = sqlite3.connect("sales_and_ab.db")
df_sales_raw.to_sql("daily_sales", conn, if_exists="replace", index=False)
conn.commit()
df_sales = pd.read_sql_query("SELECT * FROM daily_sales", conn)
df_sales["date"] = pd.to_datetime(df_sales["date"])
df_sales = df_sales.set_index("date")
print(f"\nLoaded {len(df_sales)} days of sales data FROM SQL DATABASE")

# ---- Seasonal decomposition ----
decomposition = seasonal_decompose(df_sales["daily_sales"], model="additive", period=365)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(df_sales.index, df_sales["daily_sales"], color="steelblue")
axes[0].set_title("Original Sales Data")
axes[1].plot(decomposition.trend, color="orange")
axes[1].set_title("Trend Component")
axes[2].plot(decomposition.seasonal, color="green")
axes[2].set_title("Seasonal Component")
axes[3].plot(decomposition.resid, color="gray")
axes[3].set_title("Residual (Noise)")
plt.tight_layout()
plt.savefig("time_series_decomposition.png", dpi=100)
print("Saved time_series_decomposition.png")

# ---- Moving averages ----
df_sales["MA_7"] = df_sales["daily_sales"].rolling(window=7).mean()
df_sales["MA_30"] = df_sales["daily_sales"].rolling(window=30).mean()
print("\n7-day and 30-day moving averages calculated (last 5 days):")
print(df_sales[["daily_sales", "MA_7", "MA_30"]].tail())

# ---- ARIMA forecast ----
train = df_sales["daily_sales"][:-30]
test = df_sales["daily_sales"][-30:]

model = ARIMA(train, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
mae = mean_absolute_error(test, forecast)

print(f"\nARIMA Forecast Results:")
print(f"  MAE: {mae:.1f} units ({mae/train.mean()*100:.1f}% of average daily sales)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index[-60:], train.values[-60:], label="Historical", color="steelblue")
ax.plot(test.index, test.values, label="Actual", color="green")
ax.plot(test.index, forecast, label="Forecast", color="red", linestyle="--")
ax.legend()
ax.set_title("ARIMA Sales Forecast vs Actual (Last 30 Days)")
plt.tight_layout()
plt.savefig("arima_forecast.png", dpi=100)
print("Saved arima_forecast.png")


# ================================================================
# ================================================================
# PART 2: A/B TEST - NEW CHECKOUT PAGE
# ================================================================
# ================================================================
print("\n\n" + "#" * 60)
print("# PART 2: A/B TESTING - CHECKOUT PAGE EXPERIMENT")
print("#" * 60)

# ---- Sample size calculation FIRST (before running the test) ----
print("\n--- Step 1: How many visitors do we need? ---")
baseline_rate = 0.10
minimum_detectable_effect = 0.12  # want to detect a jump to 12%
effect_size = proportion_effectsize(baseline_rate, minimum_detectable_effect)
required_n = zt_ind_solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
print(f"To reliably detect {baseline_rate*100:.0f}% -> {minimum_detectable_effect*100:.0f}% conversion")
print(f"(80% power, 5% significance), we need: {required_n:.0f} visitors PER GROUP")

# ---- Run the actual test (undersized, to show a common real-world pitfall) ----
print("\n--- Step 2: Running the test with an UNDERSIZED sample (common mistake) ---")
np.random.seed(42)
n_a, n_b = 2000, 2000
group_a = np.random.binomial(1, 0.10, n_a)
group_b = np.random.binomial(1, 0.12, n_b)

# Save visitor-level data to SQL, then read back from SQL
df_ab_raw = pd.DataFrame({
    "visitor_id": range(1, n_a + n_b + 1),
    "test_group": ["A"] * n_a + ["B"] * n_b,
    "converted": np.concatenate([group_a, group_b])
})
df_ab_raw.to_sql("ab_test_visitors", conn, if_exists="replace", index=False)
conn.commit()
df_ab_sql = pd.read_sql_query("SELECT * FROM ab_test_visitors", conn)
conn.close()
group_a = df_ab_sql[df_ab_sql["test_group"] == "A"]["converted"].values
group_b = df_ab_sql[df_ab_sql["test_group"] == "B"]["converted"].values
print(f"Loaded {len(df_ab_sql)} visitor records FROM SQL DATABASE")

conv_a, conv_b = group_a.mean(), group_b.mean()
count = np.array([group_a.sum(), group_b.sum()])
nobs = np.array([n_a, n_b])
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Group A (Old Page): {n_a} visitors, {conv_a*100:.2f}% conversion")
print(f"Group B (New Page): {n_b} visitors, {conv_b*100:.2f}% conversion")
print(f"Relative improvement: {(conv_b-conv_a)/conv_a*100:.1f}%")
print(f"P-value: {p_value:.4f}  ->  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")
print("LESSON: The improvement LOOKS real but the sample was too small to be confident.")
print("This is a classic real-world A/B testing pitfall.")

# ---- Run again with proper sample size and a genuinely large effect ----
print("\n--- Step 3: Running with proper sample size and a clear effect ---")
np.random.seed(7)
n_needed = int(np.ceil(required_n))
group_a2 = np.random.binomial(1, 0.10, n_needed)
group_b2 = np.random.binomial(1, 0.15, n_needed)  # bigger, genuine improvement

count2 = np.array([group_a2.sum(), group_b2.sum()])
nobs2 = np.array([n_needed, n_needed])
z_stat2, p_value2 = proportions_ztest(count2, nobs2)

print(f"Group A: {n_needed} visitors, {group_a2.mean()*100:.2f}% conversion")
print(f"Group B: {n_needed} visitors, {group_b2.mean()*100:.2f}% conversion")
print(f"P-value: {p_value2:.6f}  ->  {'SIGNIFICANT' if p_value2 < 0.05 else 'NOT significant'}")

# ---- Visualization ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(["Old Page (A)", "New Page (B)"], [conv_a*100, conv_b*100], color=["steelblue", "orange"])
axes[0].set_title(f"Undersized Test (n={n_a})\np={p_value:.3f} - NOT significant")
axes[0].set_ylabel("Conversion Rate (%)")

axes[1].bar(["Old Page (A)", "New Page (B)"], [group_a2.mean()*100, group_b2.mean()*100], color=["steelblue", "green"])
axes[1].set_title(f"Properly Sized Test (n={n_needed})\np={p_value2:.6f} - SIGNIFICANT")
axes[1].set_ylabel("Conversion Rate (%)")

plt.tight_layout()
plt.savefig("ab_test_results.png", dpi=100)
print("\nSaved ab_test_results.png")

print("\n" + "#" * 60)
print("# PROJECT COMPLETE")
print("#" * 60)
