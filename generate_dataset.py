"""
generate_dataset.py
--------------------
This sandbox has no internet access, so a real Kaggle House Price dataset
could not be downloaded directly. This script generates a realistic
HOUSE-PRICE-STYLE dataset with common real-estate features and a price
that is a genuine (noisy) linear function of them, so linear regression
is a meaningful, well-suited model for it.

If you have internet access, you can swap in a real dataset, e.g. the
Kaggle "House Price Prediction Dataset" from:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
Just make sure the column names in house_prices.csv match (or update the
FEATURES list at the top of linear_regression.py to match your file).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

area_sqft = np.random.normal(1800, 650, N).clip(400, 5000)
bedrooms = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.05, 0.25, 0.35, 0.25, 0.10])
bathrooms = np.clip(np.round(bedrooms * 0.6 + np.random.normal(0, 0.5, N)), 1, 4)
age_years = np.random.uniform(0, 50, N)
distance_to_city_km = np.random.exponential(8, N).clip(0.5, 60)
stories = np.random.choice([1, 2, 3], size=N, p=[0.45, 0.45, 0.10])

# True underlying linear relationship (with some noise) used to generate price.
# Price in (thousands of $):
price = (
    50
    + 0.12 * area_sqft
    + 8 * bedrooms
    + 6 * bathrooms
    - 0.8 * age_years
    - 1.5 * distance_to_city_km
    + 5 * stories
    + np.random.normal(0, 25, N)  # noise
)
price = np.clip(price, 30, None)  # no negative/unrealistic prices

df = pd.DataFrame({
    "Area_sqft": np.round(area_sqft, 1),
    "Bedrooms": bedrooms.astype(int),
    "Bathrooms": bathrooms.astype(int),
    "Age_years": np.round(age_years, 1),
    "Distance_to_City_km": np.round(distance_to_city_km, 2),
    "Stories": stories.astype(int),
    "Price_thousand_usd": np.round(price, 2),
})

df.to_csv("data/house_prices.csv", index=False)
print("Saved data/house_prices.csv with shape:", df.shape)
print(df.head())
