import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os

def load_data(x_name, y_name, data_path):
    try:
        df = pd.read_csv(data_path)
        df = df.dropna()
        
        if x_name not in df.columns or y_name not in df.columns:
            print(f"Required columns not found in {data_path}")
            return None, None
            
        X = df[[x_name]]
        y = df[y_name]
        
        return X, y
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None, None

def train_model(X, y, x_name, y_name, location_name, model_type, save_path=None):
    if model_type == 'RandomForest':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n=== {location_name} - Random Forest Regression Results ===")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        feature_importance = rf_model.feature_importances_
        print(f"\nFeature Importance:")
        print(f"Rainfall: {feature_importance[0]:.4f}")

        # Create subplots with Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted Values', 'Residual Plot', 
                           'Feature Importance', f'{x_name} vs {y_name} with Prediction Line'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Scatter plot of actual vs predicted values
        fig.add_trace(
            go.Scatter(
                x=y_test, y=y_pred, mode='markers', 
                name='Predictions', marker=dict(opacity=0.6)
            ),
            row=1, col=1
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], 
                mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text=f'Actual {y_name}', row=1, col=1)
        fig.update_yaxes(title_text=f'Predicted {y_name}', row=1, col=1)

        # 2. Residual plot
        residuals = y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode='markers', 
                name='Residuals', marker=dict(opacity=0.6)
            ),
            row=1, col=2
        )
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0], 
                mode='lines', name='Zero Line', line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text='Predicted Values', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)

        # 3. Feature importance plot
        fig.add_trace(
            go.Bar(x=[x_name], y=feature_importance, name='Feature Importance'),
            row=2, col=1
        )
        fig.update_xaxes(title_text='Features', row=2, col=1)
        fig.update_yaxes(title_text='Importance', row=2, col=1)

        # 4. Prediction line plot
        rainfall_range = np.linspace(X[x_name].min(), X[x_name].max(), 100).reshape(-1, 1)
        predictions_range = rf_model.predict(rainfall_range)

        # Actual test data
        fig.add_trace(
            go.Scatter(
                x=X_test[x_name], y=y_test, mode='markers', 
                name='Actual Test Data', marker=dict(color='blue', opacity=0.6)
            ),
            row=2, col=2
        )
        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=rainfall_range.flatten(), y=predictions_range, 
                mode='lines', name='Prediction Line', line=dict(color='red')
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text=x_name, row=2, col=2)
        fig.update_yaxes(title_text=y_name, row=2, col=2)

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text=f"{location_name} - Random Forest Regression: {x_name} vs {y_name}",
            showlegend=True,
            annotations=[
                dict(
                    x=0.02,
                    y=0.02,
                    xref='paper',
                    yref='paper',
                    text=f'MSE: {mse:.4f}<br>RMSE: {rmse:.4f}<br>R²: {r2:.4f}',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='wheat',
                    opacity=0.7,
                    align='left'
                )
            ]
        )

        # Save to HTML file if path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pyo.plot(fig, filename=save_path, auto_open=False)
            print(f"Figure saved to: {save_path}")

        # Additional: Feature importance from all trees (optional)
        print("\nFeature importance from all trees (first 5 trees):")
        for i, tree in enumerate(rf_model.estimators_[:5]):
            print(f"Tree {i+1}: {tree.feature_importances_[0]:.4f}")
            
        return mse, rmse, r2
    
    elif model_type == 'LinearRegression':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Make predictions
        y_pred = lr_model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n=== {location_name} - Linear Regression Results ===")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        coefficient = lr_model.coef_[0]
        intercept = lr_model.intercept_
        print(f"\nCoefficient: {coefficient:.4f}")
        print(f"Intercept: {intercept:.4f}")

        # Create subplots with Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted Values', 'Residual Plot', 
                           'Coefficient Value', f'{x_name} vs {y_name} with Regression Line'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Scatter plot of actual vs predicted values
        fig.add_trace(
            go.Scatter(
                x=y_test, y=y_pred, mode='markers', 
                name='Predictions', marker=dict(opacity=0.6)
            ),
            row=1, col=1
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], 
                mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text=f'Actual {y_name}', row=1, col=1)
        fig.update_yaxes(title_text=f'Predicted {y_name}', row=1, col=1)

        # 2. Residual plot
        residuals = y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode='markers', 
                name='Residuals', marker=dict(opacity=0.6)
            ),
            row=1, col=2
        )
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0], 
                mode='lines', name='Zero Line', line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text='Predicted Values', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)

        # 3. Coefficient value plot
        fig.add_trace(
            go.Bar(x=[x_name], y=[coefficient], name='Coefficient'),
            row=2, col=1
        )
        fig.update_xaxes(title_text='Features', row=2, col=1)
        fig.update_yaxes(title_text='Coefficient Value', row=2, col=1)

        # 4. Regression line plot
        # Generate points for the regression line
        x_range = np.linspace(X[x_name].min(), X[x_name].max(), 100).reshape(-1, 1)
        y_range = lr_model.predict(x_range)

        # Actual test data
        fig.add_trace(
            go.Scatter(
                x=X_test[x_name], y=y_test, mode='markers', 
                name='Actual Test Data', marker=dict(color='blue', opacity=0.6)
            ),
            row=2, col=2
        )
        # Regression line
        fig.add_trace(
            go.Scatter(
                x=x_range.flatten(), y=y_range, 
                mode='lines', name='Regression Line', line=dict(color='red')
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text=x_name, row=2, col=2)
        fig.update_yaxes(title_text=y_name, row=2, col=2)

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text=f"{location_name} - Linear Regression: {x_name} vs {y_name}",
            showlegend=True,
            annotations=[
                dict(
                    x=0.02,
                    y=0.02,
                    xref='paper',
                    yref='paper',
                    text=f'MSE: {mse:.4f}<br>RMSE: {rmse:.4f}<br>R²: {r2:.4f}<br>Coefficient: {coefficient:.4f}<br>Intercept: {intercept:.4f}',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='wheat',
                    opacity=0.7,
                    align='left'
                )
            ]
        )

        # Save to HTML file if path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pyo.plot(fig, filename=save_path, auto_open=False)
            print(f"Figure saved to: {save_path}")
            
        return mse, rmse, r2

def process_all_locations():
    base_data_dir = 'PreprocessData/elec'
    x_name = 'index'
    y_name = 'Average_MWH'
    
    # Process both Random Forest and Linear Regression
    for model_type in ['RandomForest', 'LinearRegression']:
        base_output_dir = f'html_graphs/Elec/{model_type}'
        
        # Get all subdirectories in the base data directory
        try:
            locations = [d for d in os.listdir(base_data_dir) 
                        if os.path.isdir(os.path.join(base_data_dir, d))]
        except FileNotFoundError:
            print(f"Directory {base_data_dir} not found!")
            continue
        
        results = []
        
        for location in locations:
            print(f"\n{'='*50}")
            print(f"Processing location: {location} with {model_type}")
            print(f"{'='*50}")
            
            data_path = os.path.join(base_data_dir, location, 'combined_avg.csv')
            save_path = os.path.join(base_output_dir, location, f'{model_type.lower()}_analysis_index.html')
            
            # Load data
            X, y = load_data(x_name, y_name, data_path)
            
            if X is None or y is None:
                print(f"Skipping {location} due to data loading issues")
                continue
                
            if len(X) == 0:
                print(f"Skipping {location} - no data available")
                continue
                
            # Train model and get results
            try:
                mse, rmse, r2 = train_model(X, y, x_name, y_name, location, model_type, save_path)
                results.append({
                    'Location': location,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'Samples': len(X)
                })
            except Exception as e:
                print(f"Error processing {location}: {e}")
                continue
        
        # Print summary of all results
        if results:
            print(f"\n{'='*60}")
            print(f"SUMMARY OF ALL LOCATIONS - {model_type}")
            print(f"{'='*60}")
            df_results = pd.DataFrame(results)
            print(df_results.to_string(index=False))
            
            # Save summary to CSV
            summary_path = os.path.join(base_output_dir, f'model_results_summary_{model_type}.csv')
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df_results.to_csv(summary_path, index=False)
            print(f"\nSummary saved to: {summary_path}")

# Run the processing for all locations and both models
process_all_locations()