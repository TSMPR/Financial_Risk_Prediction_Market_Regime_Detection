# User guide

1. Install Python 3.10+ and project dependencies.
2. Run `streamlit run app.py`.
3. Enter one or more valid Yahoo Finance tickers.
4. Choose lookback, volatility window, momentum window, and chronological test fraction.
5. Click **Fetch data & train models**.
6. Compare holdout metrics and confusion matrices.
7. Inspect feature importance and historical regime labels.
8. Export the JSON report and holdout prediction CSV.

If training fails because a class is absent from the training interval, increase the lookback, adjust ticker selection, or inspect the rule windows. If data retrieval fails, check connectivity, ticker spelling, and provider availability.
