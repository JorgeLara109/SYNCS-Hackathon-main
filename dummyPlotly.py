import numpy as np
import plotly.graph_objects as go

# Generate data
x = np.linspace(0, 10, 500)   # x values
y = np.sin(x)                 # sine wave

# Create figure
fig = go.Figure()

# Add sine wave trace
fig.add_trace(go.Scatter(
    x=x, 
    y=y, 
    mode='lines',
    name='Sine Wave'
))

# Customize layout
fig.update_layout(
    title="Interactive Sine Wave",
    xaxis_title="X",
    yaxis_title="sin(X)",
    template="plotly_dark"
)

# Save as HTML
fig.write_html("sine_wave.html")

print("Plot saved as sine_wave.html")
