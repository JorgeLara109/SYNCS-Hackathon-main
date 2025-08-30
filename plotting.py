import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def generate_sine_wave():
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sine Wave'))
    fig.update_layout(
        title="Interactive Sine Wave",
        xaxis_title="X",
        yaxis_title="sin(X)",
        template="plotly_dark"
    )
    
    # Save to backend folder
    output_path = Path("plots")  # make a 'plots' folder in your backend repo
    output_path.mkdir(exist_ok=True)
    file_path = output_path / "sine_wave.html"
    fig.write_html(file_path)
    
    return str(file_path)
