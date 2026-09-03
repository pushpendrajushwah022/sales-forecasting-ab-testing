# E-commerce Sales Forecasting + Checkout Page A/B Test

A two-part real-world data science project combining time series forecasting (predicting future sales) and A/B testing (evaluating whether a product change genuinely works) — two skills used constantly in Analytics/Data Science roles at product and e-commerce companies.

## Part 1: Time Series Sales Forecasting

**Problem:** Given 2 years of daily e-commerce sales, forecast the next 30 days.

**Approach:**
1. **Seasonal Decomposition** — Split the sales signal into Trend (long-term growth), Seasonality (yearly pattern, e.g., holiday peaks), and Residual (random noise)
2. **Moving Averages** — 7-day and 30-day rolling averages to smooth out daily noise and reveal the underlying pattern
3. **ARIMA Forecasting** — Trained an ARIMA(5,1,2) model on 701 days of data, forecast the final 30 days, and compared against actuals

**Result:** Forecast MAE of 89.5 units — about **6.2% of average daily sales**, meaning the model's predictions were off by roughly 6% on average, a reasonably strong result for a first forecasting model.

## Part 2: A/B Testing — New Checkout Page

**Problem:** A company wants to know if a redesigned checkout page genuinely improves conversion rate over the old one.

**Approach:**
1. **Sample Size Calculation (done BEFORE running the test)** — Calculated that detecting a jump from 10% to 12% conversion, with standard 80% power and 5% significance, requires **3,835 visitors per group**
2. **Demonstrated a Common Pitfall** — Ran the test with only 2,000 visitors per group (undersized). Result: a 16.8% *relative* improvement that **looked** promising but was **not statistically significant** (p = 0.091) — a classic real-world mistake of concluding a change "works" from an underpowered test
3. **Properly Sized Test** — Re-ran with the calculated sample size (3,835 per group) and a genuine effect (10% → 15%), which correctly came back statistically significant (p < 0.000001)

**Key lesson:** A result *looking* better is not the same as a result being *statistically reliable*. Calculating required sample size before running a test — not after seeing promising-looking numbers — is essential to avoid false positives.

## Files

- `sales_forecasting_and_ab_testing.py` — Full script covering both parts
- `time_series_decomposition.png` — Trend/seasonality/residual breakdown
- `arima_forecast.png` — 30-day forecast vs actual sales
- `ab_test_results.png` — Side-by-side comparison of the undersized vs properly-sized A/B test

## How to Run

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
python sales_forecasting_and_ab_testing.py
```

## Tech Stack

Python, Pandas, NumPy, Statsmodels (ARIMA, seasonal decomposition, power analysis), Scikit-learn, Matplotlib
