import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range for 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Generate base temperature data with seasonal variation
days = np.arange(len(dates))
avg_max_temp = 15 * np.sin(2 * np.pi * (days - 105) / 365) + 20  # Annual variation
avg_min_temp = 15 * np.sin(2 * np.pi * (days - 105) / 365) + 10

# Add noise to temperatures
max_temp = avg_max_temp + np.random.normal(0, 3, len(dates))
min_temp = avg_min_temp + np.random.normal(0, 3, len(dates))

# Generate rainfall data (more in winter, less in summer)
rain_frequency = 0.3 * np.sin(2 * np.pi * (days + 75) / 365) + 0.5
rainfall = np.zeros(len(dates))
for i in range(len(dates)):
    if np.random.random() < rain_frequency[i]:
        rainfall[i] = np.random.gamma(shape=2, scale=2)  # Gamma distribution for rainfall amount

# Generate pollution data (PM2.5)
base_pollution = 30
temp_effect = 0.5 * (20 - (max_temp + min_temp)/2)  # Increased pollution when colder
rain_effect = -0.7 * rainfall  # Rain reduces pollution
noise = np.random.normal(0, 5, len(dates))

pollution = base_pollution + temp_effect + rain_effect + noise
pollution = np.clip(pollution, 15, 100)  # Keep within realistic range

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Max_Temperature': np.round(max_temp, 1),
    'Min_Temperature': np.round(min_temp, 1),
    'Rainfall': np.round(rainfall, 1),
    'Pollution': np.round(pollution, 1)
})

# Save to CSV
df.to_csv('pollution_weather_data.csv', index=False)
print("Data saved to pollution_weather_data.csv")

# Display first few rows
print("\nGenerated data preview:")
print(df.head())