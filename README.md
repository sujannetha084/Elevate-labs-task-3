# Task 3: Linear Regression
**AI & ML Internship — Elevate Labs**

## Objective
Implement and understand simple & multiple linear regression.

## Tools Used
Python, Scikit-learn, Pandas, Matplotlib, Seaborn

## Dataset
House Price dataset (`data/house_prices.csv`) with features: `Area_sqft`,
`Bedrooms`, `Bathrooms`, `Age_years`, `Distance_to_City_km`, `Stories`, and
target `Price_thousand_usd`.

> **Note on the data file:** generated locally with `generate_dataset.py`
> since this environment has no internet access to download a real
> dataset. Price was built as a genuine (noisy) linear combination of the
> features, so it's a fair, realistic dataset for demonstrating linear
> regression end-to-end. If you'd like to use a real dataset instead, the
> [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
> works well — just update the `FEATURES`/`TARGET` variables at the top of
> `linear_regression.py` to match its column names.

## Project Structure
```
├── data/
│   └── house_prices.csv
├── images/
│   ├── simple_regression_line.png
│   ├── multiple_regression_actual_vs_predicted.png
│   ├── multiple_regression_coefficients.png
│   └── residual_plot.png
├── generate_dataset.py
├── linear_regression.py
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python generate_dataset.py       # only needed if data/house_prices.csv doesn't exist
python linear_regression.py
```

## What Was Done

1. **Loaded and preprocessed** the dataset — checked shape, nulls, and
   summary statistics. All features were already numeric, so no encoding
   was needed here (see Task 1 for the full cleaning/encoding workflow).
2. **Train-test split** — 80/20 split using `train_test_split`.
3. **Fit two models** with `sklearn.linear_model.LinearRegression`:
   - **Simple Linear Regression:** `Price ~ Area_sqft` (1 feature)
   - **Multiple Linear Regression:** `Price ~` all 6 features
4. **Evaluated both** with MAE, MSE, RMSE, and R².
5. **Plotted and interpreted:**
   - Regression line over the test set (simple model)
   - Actual vs. Predicted scatter (multiple model)
   - Coefficient bar chart (multiple model) to see each feature's direction/size of effect
   - Residual plot to visually sanity-check the regression assumptions

## Results

| Model                     | MAE   | MSE     | RMSE  | R²    |
|---------------------------|-------|---------|-------|-------|
| Simple (Area only)        | 28.30 | 1179.72 | 34.35 | 0.826 |
| Multiple (all features)   | 19.79 | 567.17  | 23.82 | 0.916 |

(Exact numbers are in `model_comparison.csv`, regenerated fresh each run.)

**Coefficient interpretation (multiple model):** holding all other features
fixed, `Bathrooms` and `Bedrooms` have the largest positive effect on price
per unit increase, `Distance_to_City_km` and `Age_years` push price down,
and `Area_sqft`'s effect looks small per-unit only because it's measured
in single square feet rather than larger units (its total contribution
across a typical house's full area is still substantial).

**Simple vs. multiple:** adding the extra features raised R² from 0.826 to
0.916 and lowered every error metric — the additional predictors do
capture a real, useful chunk of the variance in price beyond area alone.

---

## Interview Questions & Answers

**1. What assumptions does linear regression make?**
- **Linearity:** the relationship between predictors and the target is linear.
- **Independence:** observations (and residuals) are independent of each other.
- **Homoscedasticity:** residuals have constant variance across all predicted values (no "fanning out").
- **Normality of residuals:** residuals are approximately normally distributed (mainly matters for inference/confidence intervals, less for pure prediction).
- **No (or low) multicollinearity:** predictors aren't highly correlated with each other.

**2. How do you interpret the coefficients?**
Each coefficient represents the expected change in the target variable for
a one-unit increase in that feature, **holding all other features
constant** (in multiple regression). The sign tells you the direction of
the relationship, and the magnitude (relative to the feature's scale)
tells you the strength. The intercept is the predicted value when all
features are zero (which may or may not be a meaningful real-world point).

**3. What is R² score and its significance?**
R² (coefficient of determination) measures the proportion of variance in
the target variable that is explained by the model, ranging from 0 to 1
(it can go negative for a very poor fit on new data). An R² of 0.92 means
the model explains 92% of the variance in price. It's useful for
comparing models on the same dataset, but a high R² alone doesn't
guarantee a good model — it can be inflated by overfitting or by simply
adding more features (adjusted R² corrects for that).

**4. When would you prefer MSE over MAE?**
MSE (Mean Squared Error) squares the errors, so it penalizes large errors
much more heavily than small ones — prefer it when large mistakes are
disproportionately costly and you want the model to focus on avoiding
them. MAE (Mean Absolute Error) treats all error sizes proportionally and
is more robust to outliers, so it's preferable when you want a metric
that's easier to interpret in the original units and isn't dominated by a
few extreme errors.

**5. How do you detect multicollinearity?**
- A **correlation matrix** between predictors — very high pairwise
  correlations (e.g. >0.8) are a red flag.
- The **Variance Inflation Factor (VIF)** for each predictor — a VIF above
  ~5-10 typically indicates problematic multicollinearity.
- Watching for **unstable or counter-intuitive coefficients** (e.g. a
  coefficient with an unexpected sign, or one that swings wildly when a
  similar feature is added/removed).

**6. What is the difference between simple and multiple regression?**
**Simple linear regression** models the relationship between a target
variable and a **single** predictor. **Multiple linear regression**
extends this to **two or more** predictors, allowing the model to account
for several factors simultaneously and isolate each one's individual
effect (holding the others constant), typically improving accuracy over
a single-feature model when the extra features genuinely add signal (as
seen in the results table above).

**7. Can linear regression be used for classification?**
Not directly and not well — linear regression predicts continuous, unbounded
values, so it doesn't naturally output probabilities or produce a clean
decision boundary for class labels. Using it directly on a 0/1 target can
produce predictions outside [0, 1] and is sensitive to outliers. For
classification, **logistic regression** is the appropriate adaptation — it
applies a sigmoid function to a linear combination of inputs so the output
is a proper probability between 0 and 1.

**8. What happens if you violate regression assumptions?**
- **Non-linearity:** the model will systematically under/over-predict in
  certain ranges — a straight line just can't capture the true pattern.
- **Non-independence** (e.g. time-series autocorrelation): standard errors
  and significance tests become unreliable, even if the coefficients
  themselves aren't badly biased.
- **Heteroscedasticity** (non-constant variance): coefficient estimates
  stay unbiased, but standard errors/p-values become unreliable, hurting
  inference (visible as a "fanning" pattern in the residual plot).
- **Multicollinearity:** coefficients become unstable and hard to
  interpret, though overall predictive accuracy may still be fine.
- In general, the model can still run and produce *a* prediction, but
  the coefficients' interpretability and the model's statistical validity
  suffer, and predictive accuracy can degrade on new data.
