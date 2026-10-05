"""
Task 3: Linear Regression
AI & ML Internship - Elevate Labs

Objective: Implement and understand simple & multiple linear regression.
Dataset: House Price dataset (data/house_prices.csv)

Steps covered:
1. Import and preprocess the dataset
2. Split data into train-test sets
3. Fit a Linear Regression model using sklearn.linear_model
4. Evaluate model using MAE, MSE, R^2
5. Plot regression line and interpret coefficients

This script runs TWO models:
  A) Simple Linear Regression  -> Price ~ Area_sqft   (1 feature)
  B) Multiple Linear Regression -> Price ~ all features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

TARGET = "Price_thousand_usd"
FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Distance_to_City_km", "Stories"]

# ---------------------------------------------------------------
# STEP 1: Import and preprocess the dataset
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOAD & PREPROCESS")
print("=" * 60)

df = pd.read_csv("data/house_prices.csv")
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# No missing values / categorical columns in this dataset by construction,
# but in general this is where you'd impute nulls and encode categoricals
# (see Task 1 for that full workflow). All features here are already numeric.

# ---------------------------------------------------------------
# PART A: SIMPLE LINEAR REGRESSION (Price ~ Area_sqft)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PART A: SIMPLE LINEAR REGRESSION (Price ~ Area_sqft)")
print("=" * 60)

X_simple = df[["Area_sqft"]]
y = df[TARGET]

# STEP 2: Train-test split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train_s.shape[0]}, Test size: {X_test_s.shape[0]}")

# STEP 3: Fit model
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# STEP 4: Evaluate
y_pred_s = simple_model.predict(X_test_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mse_s)
r2_s = r2_score(y_test_s, y_pred_s)

print(f"\nIntercept: {simple_model.intercept_:.3f}")
print(f"Coefficient (Area_sqft): {simple_model.coef_[0]:.4f}")
print(f"\nMAE:  {mae_s:.3f}")
print(f"MSE:  {mse_s:.3f}")
print(f"RMSE: {rmse_s:.3f}")
print(f"R^2:  {r2_s:.4f}")

print(
    f"\nInterpretation: for every extra 1 sq.ft. of area, predicted price "
    f"increases by about ${simple_model.coef_[0]*1000:.1f} "
    f"(coefficient is in thousands of USD)."
)

# STEP 5: Plot regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test_s, y_test_s, color="steelblue", alpha=0.6, label="Actual (test set)")
# sort for a clean line
order = X_test_s["Area_sqft"].argsort()
plt.plot(
    X_test_s["Area_sqft"].values[order],
    y_pred_s[order],
    color="red",
    linewidth=2,
    label="Regression line",
)
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price (thousand USD)")
plt.title(f"Simple Linear Regression: Price vs Area (R^2 = {r2_s:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("images/simple_regression_line.png", dpi=120)
plt.close()
print("\nSaved images/simple_regression_line.png")

# ---------------------------------------------------------------
# PART B: MULTIPLE LINEAR REGRESSION (Price ~ all features)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: MULTIPLE LINEAR REGRESSION (Price ~ all features)")
print("=" * 60)

X_multi = df[FEATURES]

# STEP 2: Train-test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train_m.shape[0]}, Test size: {X_test_m.shape[0]}")

# STEP 3: Fit model
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# STEP 4: Evaluate
y_pred_m = multi_model.predict(X_test_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = np.sqrt(mse_m)
r2_m = r2_score(y_test_m, y_pred_m)

print(f"\nIntercept: {multi_model.intercept_:.3f}")
coef_table = pd.DataFrame({"Feature": FEATURES, "Coefficient": multi_model.coef_})
print("\nCoefficients:\n", coef_table.to_string(index=False))

print(f"\nMAE:  {mae_m:.3f}")
print(f"MSE:  {mse_m:.3f}")
print(f"RMSE: {rmse_m:.3f}")
print(f"R^2:  {r2_m:.4f}")

print(
    "\nInterpretation: holding all other features constant, each coefficient shows "
    "the expected change in Price (in thousand USD) for a one-unit increase in that "
    "feature. E.g. Area_sqft's coefficient shows the price impact of 1 extra sq.ft. "
    "of area while bedrooms/bathrooms/age/distance/stories stay fixed."
)

# STEP 5: Actual vs Predicted plot (standard way to visualize multi-feature regression fit)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_m, y_pred_m, color="seagreen", alpha=0.6)
lims = [min(y_test_m.min(), y_pred_m.min()), max(y_test_m.max(), y_pred_m.max())]
plt.plot(lims, lims, color="red", linewidth=2, label="Perfect prediction")
plt.xlabel("Actual Price (thousand USD)")
plt.ylabel("Predicted Price (thousand USD)")
plt.title(f"Multiple Linear Regression: Actual vs Predicted (R^2 = {r2_m:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("images/multiple_regression_actual_vs_predicted.png", dpi=120)
plt.close()
print("Saved images/multiple_regression_actual_vs_predicted.png")

# Coefficient bar chart
plt.figure(figsize=(8, 5))
colors = ["seagreen" if c >= 0 else "indianred" for c in multi_model.coef_]
plt.barh(FEATURES, multi_model.coef_, color=colors)
plt.xlabel("Coefficient value (impact on Price, thousand USD)")
plt.title("Multiple Linear Regression Coefficients")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("images/multiple_regression_coefficients.png", dpi=120)
plt.close()
print("Saved images/multiple_regression_coefficients.png")

# Residuals plot (checking regression assumptions)
residuals = y_test_m - y_pred_m
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_m, residuals, color="darkorange", alpha=0.6)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Predicted Price (thousand USD)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (checking homoscedasticity assumption)")
plt.tight_layout()
plt.savefig("images/residual_plot.png", dpi=120)
plt.close()
print("Saved images/residual_plot.png")

# ---------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: SIMPLE vs MULTIPLE REGRESSION")
print("=" * 60)
summary = pd.DataFrame({
    "Model": ["Simple (Area only)", "Multiple (all features)"],
    "MAE": [mae_s, mae_m],
    "MSE": [mse_s, mse_m],
    "RMSE": [rmse_s, rmse_m],
    "R2": [r2_s, r2_m],
})
print("\n", summary.to_string(index=False))
summary.to_csv("model_comparison.csv", index=False)
print("\nSaved model_comparison.csv")
