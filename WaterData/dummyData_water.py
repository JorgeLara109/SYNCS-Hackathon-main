import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of days in a year
days = 365
dates = pd.date_range("2024-01-01", periods=days)

# Assume one city with constant population density
population_density = 5000  # people/km^2

# Generate seasonal temperatures using sine waves
day_of_year = np.arange(days)
max_temp = 30 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, days)
min_temp = 15 + 8 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, days)

# Water usage model (liters per capita per day)
# Base + effect of density + temperature effects
water_usage = (
    100          # base usage
    + 0.01 * population_density
    + 2 * (max_temp - 20)    # more usage when hotter
    - 1 * (15 - min_temp)    # more usage when colder nights (heating)
    + np.random.normal(0, 10, days)  # random noise
)

# Ensure water usage is not negative
water_usage = np.clip(water_usage, 50, None)

# Build DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Population_Density": population_density,
    "Max_Temperature_C": max_temp.round(2),
    "Min_Temperature_C": min_temp.round(2),
    "Water_Usage_Liters_per_Capita": water_usage.round(1)
})

# Save to CSV
df.to_csv("water_usage_data.csv", index=False)

print("Data saved to water_usage_data.csv")
print(df.head())
