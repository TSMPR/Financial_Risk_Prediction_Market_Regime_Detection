# Methodology

## Data handling
Adjusted prices are sorted, duplicate dates removed, non-numeric/non-positive values treated as missing, short gaps forward-filled up to three sessions, and common complete observations retained.

## Feature timing
At date t, features include one-day return, trailing momentum, trailing rolling volatility, equal-weight market return/momentum/volatility, and drawdown. They use data through t only.

## Target timing and labels
The target is a forward-looking outcome. Forward realized volatility is calculated over the next volatility-window sessions. The high-risk cutoff is the expanding historical 70th percentile of trailing realized volatility, requiring at least 60 observations. Remaining observations are categorized by trailing momentum: above +3% bullish, below -3% bearish, otherwise sideways. Initial and terminal rows without sufficient data are omitted.

## Chronological evaluation
The data are sorted by time and split without shuffling. All imputers and scalers are inside model pipelines and fit only on training rows. Balanced class weights are used for Logistic Regression and Random Forest; balanced sample weights are used for HistGradientBoosting.

## Models
- Logistic Regression: standardized features, balanced classes.
- Random Forest: 350 trees, balanced subsampling, minimum leaf size 3.
- HistGradientBoosting: 150 boosting iterations, regularization, balanced training sample weights.

## Metrics
Accuracy and weighted precision, recall, and F1 are calculated on the holdout. Confusion matrix labels are held constant across models.

## Interpretation
Because targets are rule-derived, reported scores measure how well models reproduce the defined heuristic on later observations. They do not establish that market regimes are objectively known or that future market movements are predictable.
