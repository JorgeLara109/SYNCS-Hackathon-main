import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Create directory for saving plots if it doesn't exist
os.makedirs('html_dummyPol', exist_ok=True)

# Generate synthetic data
def generate_pollution_data():
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
    
    return df

# Generate and save the data
df = generate_pollution_data()
df.to_csv('pollution_weather_data.csv', index=False)
print("Data generated and saved to pollution_weather_data.csv")

# Prepare data for modeling
X = df[['Min_Temperature']]
y = df['Pollution']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Create visualization 1: Actual vs Predicted values with residuals
fig1 = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Actual vs Predicted Pollution Levels', 'Residuals Plot'),
                    vertical_spacing=0.15)

# Top plot: Actual vs Predicted
fig1.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                         name='Predictions',
                         marker=dict(color='blue', opacity=0.6)),
              row=1, col=1)
fig1.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                         mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')),
              row=1, col=1)

# Bottom plot: Residuals
fig1.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                         name='Residuals',
                         marker=dict(color='green', opacity=0.6)),
              row=2, col=1)
fig1.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0], 
                         mode='lines', name='Zero Residual', line=dict(color='red', dash='dash')),
              row=2, col=1)

fig1.update_layout(height=800, title_text="Prediction Analysis with Residuals", showlegend=True)
fig1.update_xaxes(title_text="Actual Pollution (µg/m³)", row=1, col=1)
fig1.update_yaxes(title_text="Predicted Pollution (µg/m³)", row=1, col=1)
fig1.update_xaxes(title_text="Predicted Values", row=2, col=1)
fig1.update_yaxes(title_text="Residuals", row=2, col=1)

fig1.write_html('html_dummyPol/actual_vs_predicted_with_residuals.html')

# Create visualization 2: Feature importance with detailed analysis
feature_importance = pd.DataFrame({
    'feature': ['Min_Temperature'],
    'importance': rf_model.feature_importances_
})

fig2 = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Feature Importance', 'Feature vs Target'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}]])

# Left plot: Feature importance
fig2.add_trace(go.Bar(x=feature_importance['feature'], y=feature_importance['importance'],
                     marker_color='lightblue'),
              row=1, col=1)

# Right plot: Feature vs Target
fig2.add_trace(go.Scatter(x=df['Min_Temperature'], y=df['Pollution'], mode='markers',
                         marker=dict(color=df['Min_Temperature'], colorscale='viridis', showscale=True,
                                   colorbar=dict(title='Min Temperature (°C)')),
                         name='Temperature vs Pollution'),
              row=1, col=2)

# Add trend line
z = np.polyfit(df['Min_Temperature'], df['Pollution'], 1)
p = np.poly1d(z)
fig2.add_trace(go.Scatter(x=df['Min_Temperature'], y=p(df['Min_Temperature']), mode='lines',
                         line=dict(color='red', width=2), name='Trend Line'),
              row=1, col=2)

fig2.update_layout(height=500, title_text="Feature Analysis", showlegend=True)
fig2.update_xaxes(title_text="Features", row=1, col=1)
fig2.update_yaxes(title_text="Importance", row=1, col=1)
fig2.update_xaxes(title_text="Min Temperature (°C)", row=1, col=2)
fig2.update_yaxes(title_text="Pollution Level (µg/m³)", row=1, col=2)

fig2.write_html('html_dummyPol/feature_analysis.html')

# Create visualization 3: Relationship between Min Temperature and Pollution with residuals
fig3 = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Temperature vs Pollution', 'Residuals vs Temperature'),
                    vertical_spacing=0.15)

# Top plot: Temperature vs Pollution
fig3.add_trace(go.Scatter(x=df['Min_Temperature'], y=df['Pollution'], mode='markers',
                         marker=dict(color=df['Min_Temperature'], colorscale='viridis', 
                                   showscale=True, colorbar=dict(title='Min Temperature (°C)')),
                         name='Actual Data'),
              row=1, col=1)

# Create prediction line
min_temp_range = np.linspace(df['Min_Temperature'].min(), df['Min_Temperature'].max(), 100).reshape(-1, 1)
pollution_pred = rf_model.predict(min_temp_range)
fig3.add_trace(go.Scatter(x=min_temp_range.flatten(), y=pollution_pred, mode='lines',
                         line=dict(color='red', width=3), name='Model Prediction'),
              row=1, col=1)

# Bottom plot: Residuals vs Temperature
fig3.add_trace(go.Scatter(x=X_test['Min_Temperature'], y=residuals, mode='markers',
                         marker=dict(color=residuals, colorscale='RdBu_r', showscale=True,
                                   colorbar=dict(title='Residuals')),
                         name='Residuals'),
              row=2, col=1)
fig3.add_trace(go.Scatter(x=[X_test['Min_Temperature'].min(), X_test['Min_Temperature'].max()], y=[0, 0], 
                         mode='lines', name='Zero Residual', line=dict(color='red', dash='dash')),
              row=2, col=1)

fig3.update_layout(height=800, title_text="Temperature-Pollution Relationship with Residuals", showlegend=True)
fig3.update_xaxes(title_text="Minimum Temperature (°C)", row=1, col=1)
fig3.update_yaxes(title_text="Pollution Level (µg/m³)", row=1, col=1)
fig3.update_xaxes(title_text="Minimum Temperature (°C)", row=2, col=1)
fig3.update_yaxes(title_text="Residuals", row=2, col=1)

fig3.write_html('html_dummyPol/temperature_vs_pollution_with_residuals.html')

# Create visualization 4: Time series of pollution and temperature with prediction residuals
# Predict for all data
all_predictions = rf_model.predict(X)
all_residuals = y - all_predictions

fig4 = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Time Series of Pollution and Temperature', 'Time Series of Residuals'),
                    vertical_spacing=0.15,
                    specs=[[{"secondary_y": True}], [{}]])

# Top plot: Time series with two y-axes
fig4.add_trace(
    go.Scatter(x=df['Date'], y=df['Pollution'], name="Actual Pollution", line=dict(color='red')),
    row=1, col=1, secondary_y=False,
)
fig4.add_trace(
    go.Scatter(x=df['Date'], y=all_predictions, name="Predicted Pollution", line=dict(color='orange', dash='dash')),
    row=1, col=1, secondary_y=False,
)
fig4.add_trace(
    go.Scatter(x=df['Date'], y=df['Min_Temperature'], name="Min Temperature", line=dict(color='blue')),
    row=1, col=1, secondary_y=True,
)

# Bottom plot: Residuals over time
fig4.add_trace(
    go.Scatter(x=df['Date'], y=all_residuals, name="Residuals", line=dict(color='green')),
    row=2, col=1
)
fig4.add_trace(
    go.Scatter(x=[df['Date'].min(), df['Date'].max()], y=[0, 0], 
              mode='lines', name='Zero Residual', line=dict(color='red', dash='dash')),
    row=2, col=1
)

fig4.update_layout(height=800, title_text="Time Series Analysis with Residuals", showlegend=True)
fig4.update_xaxes(title_text="Date", row=2, col=1)
fig4.update_yaxes(title_text="Pollution (µg/m³)", row=1, col=1, secondary_y=False)
fig4.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=True)
fig4.update_yaxes(title_text="Residuals", row=2, col=1)

fig4.write_html('html_dummyPol/time_series_with_residuals.html')

