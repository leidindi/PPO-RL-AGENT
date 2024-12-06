import pickle
import os
import numpy as np
import plotly.graph_objects as go

# Function to load data and create an interactive 3D scatter viewer
def interactive_plot_value_function(data_dir=".\\value_data"):
    # Load all saved episodes
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])

    # Initialize the figure
    fig = go.Figure()

    # Load each file and add it as a frame
    frames = []
    for i, file in enumerate(files):
        with open(os.path.join(data_dir, file), "rb") as f:
            xyz = pickle.load(f)

        # Extract x, y, z
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Add data as a frame for the slider
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8)
            )],
            name=f"Episode {i * 50}"  # Assuming 50-episode intervals
        ))

    # Add the first scatter plot as the initial plot
    first_xyz = pickle.load(open(os.path.join(data_dir, files[0]), "rb"))
    fig.add_trace(go.Scatter3d(
        x=first_xyz[:, 0],
        y=first_xyz[:, 1],
        z=first_xyz[:, 2],
        mode='markers',
        marker=dict(size=3, color=first_xyz[:, 2], colorscale='Viridis', opacity=0.8)
    ))

    # Update layout with slider
    fig.update_layout(
        title="Value Function Evolution",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Value",
            zaxis=dict(range=[-10, 1])  # Freeze z-axis range
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False))])]
            )],
        sliders=[{
            "steps": [
                {"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": frame.name,
                 "method": "animate"} for frame in frames
            ],
            "currentvalue": {"font": {"size": 20}, "prefix": "Episode: ", "visible": True},
            "pad": {"t": 50},
        }]
    )

    # Add frames to the figure
    fig.frames = frames

    # Show the figure
    fig.show()

# Example usage:
interactive_plot_value_function()
