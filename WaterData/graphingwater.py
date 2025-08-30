import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import os

# Load the data
df = pd.read_csv("water_usage_data.csv")

# Features and target
X = df[["Population_Density", "Max_Temperature_C", "Min_Temperature_C"]]
y = df["Water_Usage_Liters_per_Capita"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train kNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Ensure folder exists
output_folder = "html_water"
os.makedirs(output_folder, exist_ok=True)

# Plot predicted vs actual
fig1 = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Water Usage', 'y':'Predicted Water Usage'},
                  title="Predicted vs Actual Water Usage (kNN)")
fig1.add_shape(
    type="line",
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max(),
    line=dict(color="red", dash="dash")
)
fig1.write_html(os.path.join(output_folder, "predicted_vs_actual.html"))

# Plot residuals
fig2 = px.histogram(residuals, nbins=30, marginal="box",
                    labels={'value':'Residuals'},
                    title="Residuals of kNN Predictions")
fig2.write_html(os.path.join(output_folder, "residuals.html"))

# Feature importance using permutation importance
perm_importance = permutation_importance(knn, X_test, y_test, n_repeats=30, random_state=42)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
              title="Feature Importance (Permutation Importance)")
fig3.write_html(os.path.join(output_folder, "feature_importance.html"))

print("kNN model trained and interactive Plotly HTML plots saved in 'html_water' folder.")
