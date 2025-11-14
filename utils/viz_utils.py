"""
Visualization utilities for plotting inputs, outputs, and 3D trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Optional


def plot_input_output(
    input_seq: np.ndarray,
    output_seq: np.ndarray,
    pulse_times: List[float],
    dt: float
) -> plt.Figure:
    """
    Create 2D plot of input and output signals with beat markers.

    Args:
        input_seq: Input sequence (timesteps,)
        output_seq: Output sequence (timesteps,)
        pulse_times: List of pulse onset times in seconds
        dt: Time step in seconds

    Returns:
        Matplotlib figure
    """
    # Create time axis
    n_timesteps = len(input_seq)
    time_axis = np.arange(n_timesteps) * dt

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot input
    ax1.plot(time_axis, input_seq, 'b-', linewidth=1.5, label='Input')
    ax1.set_ylabel('Input Amplitude', fontsize=10)
    ax1.set_ylim(-0.1, 0.6)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Add beat markers on input
    for i, pulse_time in enumerate(pulse_times):
        if pulse_time < time_axis[-1]:
            # Find the value at pulse time
            pulse_idx = int(pulse_time / dt)
            if pulse_idx < len(input_seq):
                ax1.scatter(pulse_time, input_seq[pulse_idx],
                          c='red', s=100, marker='*',
                          edgecolors='darkred', linewidths=2, zorder=5)

    # Add inter-pulse intervals on input plot
    sorted_pulses = sorted(pulse_times)
    for i in range(len(sorted_pulses) - 1):
        if sorted_pulses[i+1] < time_axis[-1]:
            # Calculate interval
            interval = sorted_pulses[i+1] - sorted_pulses[i]

            # Find midpoint for label placement
            mid_time = (sorted_pulses[i] + sorted_pulses[i+1]) / 2

            # Draw arrow and label
            ax1.annotate('', xy=(sorted_pulses[i+1], 0.45), xytext=(sorted_pulses[i], 0.45),
                        arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
            ax1.text(mid_time, 0.48, f'{interval:.3f}s',
                    ha='center', va='bottom', fontsize=8, color='gray')

    # Plot output
    ax2.plot(time_axis, output_seq, 'g-', linewidth=1.5, label='Output')
    ax2.set_ylabel('Output Amplitude', fontsize=10)
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Add beat markers on output
    for i, pulse_time in enumerate(pulse_times):
        if pulse_time < time_axis[-1]:
            # Find the value at pulse time
            pulse_idx = int(pulse_time / dt)
            if pulse_idx < len(output_seq):
                ax2.scatter(pulse_time, output_seq[pulse_idx],
                          c='red', s=100, marker='*',
                          edgecolors='darkred', linewidths=2, zorder=5)

    # Add pulse numbers
    ax1.set_title('Input and Output Signals with Beat Markers and Inter-Pulse Intervals', fontsize=12)

    plt.tight_layout()
    return fig


def create_3d_trajectory(
    states_pca: np.ndarray,
    pulse_times: Optional[List[float]] = None,
    dt: float = 0.001,
    title: str = "State Trajectory",
    input_seq: Optional[np.ndarray] = None,
    output_seq: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create interactive 3D trajectory plot with Plotly.

    Args:
        states_pca: PCA-transformed states (timesteps, 3)
        pulse_times: Optional list of pulse onset times
        dt: Time step in seconds
        title: Plot title
        input_seq: Optional input sequence for hover display
        output_seq: Optional output sequence for hover display

    Returns:
        Plotly figure object
    """
    # Create time-based colors
    timesteps = len(states_pca)
    time_axis = np.arange(timesteps) * dt

    # Prepare hover text with input/output values
    hover_texts = []
    for i in range(timesteps):
        hover_text = f'Time: {time_axis[i]:.3f}s<br>'
        hover_text += f'PC1: {states_pca[i, 0]:.3f}<br>'
        hover_text += f'PC2: {states_pca[i, 1]:.3f}<br>'
        hover_text += f'PC3: {states_pca[i, 2]:.3f}'

        if input_seq is not None and i < len(input_seq):
            hover_text += f'<br>Input: {input_seq[i]:.3f}'

        if output_seq is not None and i < len(output_seq):
            hover_text += f'<br>Output: {output_seq[i]:.3f}'

        hover_texts.append(hover_text)

    # Create main trajectory trace
    trajectory_trace = go.Scatter3d(
        x=states_pca[:, 0],
        y=states_pca[:, 1],
        z=states_pca[:, 2],
        mode='lines+markers',
        line=dict(
            color=time_axis,
            colorscale='Viridis',
            width=3,
            colorbar=dict(
                title='Time (s)',
                thickness=15,
                len=0.5,
                x=1.02
            )
        ),
        marker=dict(
            size=2,
            color=time_axis,
            colorscale='Viridis',
            showscale=False
        ),
        name='Trajectory',
        hovertemplate='%{text}',
        text=hover_texts
    )

    # Create figure
    fig = go.Figure(data=[trajectory_trace])

    # Add start point
    start_hover = f'START<br>Time: 0.000s<br>PC1: {states_pca[0, 0]:.3f}<br>PC2: {states_pca[0, 1]:.3f}<br>PC3: {states_pca[0, 2]:.3f}'
    if input_seq is not None:
        start_hover += f'<br>Input: {input_seq[0]:.3f}'
    if output_seq is not None:
        start_hover += f'<br>Output: {output_seq[0]:.3f}'

    fig.add_trace(go.Scatter3d(
        x=[states_pca[0, 0]],
        y=[states_pca[0, 1]],
        z=[states_pca[0, 2]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Start',
        showlegend=True,
        hovertemplate='%{text}',
        text=[start_hover]
    ))

    # Add end point
    end_idx = len(states_pca) - 1
    end_hover = f'END<br>Time: {time_axis[end_idx]:.3f}s<br>PC1: {states_pca[end_idx, 0]:.3f}<br>PC2: {states_pca[end_idx, 1]:.3f}<br>PC3: {states_pca[end_idx, 2]:.3f}'
    if input_seq is not None and end_idx < len(input_seq):
        end_hover += f'<br>Input: {input_seq[end_idx]:.3f}'
    if output_seq is not None and end_idx < len(output_seq):
        end_hover += f'<br>Output: {output_seq[end_idx]:.3f}'

    fig.add_trace(go.Scatter3d(
        x=[states_pca[-1, 0]],
        y=[states_pca[-1, 1]],
        z=[states_pca[-1, 2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='square'),
        name='End',
        showlegend=True,
        hovertemplate='%{text}',
        text=[end_hover]
    ))

    # Add beat markers if provided
    if pulse_times is not None:
        beat_indices = []
        beat_points = []
        beat_hovers = []

        for i, pulse_time in enumerate(pulse_times):
            pulse_idx = int(pulse_time / dt)
            if 0 <= pulse_idx < len(states_pca):
                beat_indices.append(pulse_idx)
                beat_points.append(states_pca[pulse_idx])

                # Create hover text for beat
                beat_hover = f'BEAT {i+1}<br>Time: {pulse_time:.3f}s<br>'
                beat_hover += f'PC1: {states_pca[pulse_idx, 0]:.3f}<br>'
                beat_hover += f'PC2: {states_pca[pulse_idx, 1]:.3f}<br>'
                beat_hover += f'PC3: {states_pca[pulse_idx, 2]:.3f}'

                if input_seq is not None and pulse_idx < len(input_seq):
                    beat_hover += f'<br>Input: {input_seq[pulse_idx]:.3f}'

                if output_seq is not None and pulse_idx < len(output_seq):
                    beat_hover += f'<br>Output: {output_seq[pulse_idx]:.3f}'

                beat_hovers.append(beat_hover)

        if beat_points:
            beat_points = np.array(beat_points)

            # Add beat markers
            fig.add_trace(go.Scatter3d(
                x=beat_points[:, 0],
                y=beat_points[:, 1],
                z=beat_points[:, 2],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                name=f'Beats (n={len(beat_points)})',
                showlegend=True,
                hovertemplate='%{text}',
                text=beat_hovers
            ))

            # Add vertical lines from beats to base plane
            for i, (idx, point) in enumerate(zip(beat_indices, beat_points)):
                z_min = states_pca[:, 2].min()
                fig.add_trace(go.Scatter3d(
                    x=[point[0], point[0]],
                    y=[point[1], point[1]],
                    z=[point[2], z_min],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='auto'
        ),
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )

    return fig


def plot_trajectory_comparison(
    states_list: List[np.ndarray],
    labels: List[str],
    title: str = "Trajectory Comparison"
) -> go.Figure:
    """
    Plot multiple trajectories for comparison.

    Args:
        states_list: List of PCA-transformed states arrays
        labels: List of labels for each trajectory
        title: Plot title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (states, label) in enumerate(zip(states_list, labels)):
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode='lines',
            line=dict(color=color, width=2),
            name=label,
            opacity=0.7
        ))

        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[states[0, 0]],
            y=[states[0, 1]],
            z=[states[0, 2]],
            mode='markers',
            marker=dict(size=8, color=color),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=700,
        showlegend=True
    )

    return fig
