"""
Interactive visualizations for LSTM dynamics analysis.
Creates dynamic plots with sliders for exploring states across epochs and periods.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple


class ErrorProgressionPlotter:
    """
    Interactive plot showing MSE vs Period with epoch slider.
    """

    def __init__(self, accuracy_file: Path):
        """
        Initialize error progression plotter.

        Args:
            accuracy_file: Path to accuracy_progression.json
        """
        with open(accuracy_file, 'r') as f:
            self.data = json.load(f)

        # Extract epochs and sort them
        self.epochs = sorted([int(e) for e in self.data.keys()])
        self.periods = self.data[str(self.epochs[0])]['periods']

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)

        # Initial plot
        self.line, = self.ax.plot([], [], 'bo-', linewidth=2, markersize=8)
        self.ax.set_xlabel('Period (s)', fontsize=12)
        self.ax.set_ylabel('Mean Squared Error', fontsize=12)
        self.ax.grid(True, alpha=0.3)

        # Setup slider
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.slider = Slider(
            ax_slider, 'Epoch',
            self.epochs[0], self.epochs[-1],
            valinit=self.epochs[0],
            valstep=1
        )
        self.slider.on_changed(self.update_plot)

        # Initialize plot
        self.update_plot(self.epochs[0])

    def update_plot(self, epoch):
        """Update plot for selected epoch."""
        # Find closest epoch
        epoch = min(self.epochs, key=lambda x: abs(x - epoch))

        if str(epoch) in self.data:
            mse_values = self.data[str(epoch)]['mse_per_period']
            periods = self.data[str(epoch)]['periods']

            # Update line data
            self.line.set_data(periods, mse_values)

            # Update y-axis limits
            self.ax.set_ylim(0, max(mse_values) * 1.1)
            self.ax.set_xlim(min(periods) - 0.05, max(periods) + 0.05)

            # Update title
            mean_mse = np.mean(mse_values)
            self.ax.set_title(
                f'Performance at Epoch {epoch} | Mean MSE: {mean_mse:.6f}',
                fontsize=14
            )

            # Clear previous annotations
            for txt in self.ax.texts:
                txt.remove()

            # Add value labels
            for period, mse in zip(periods, mse_values):
                self.ax.annotate(f'{mse:.4f}',
                               xy=(period, mse),
                               xytext=(0, 5),
                               textcoords='offset points',
                               fontsize=8,
                               ha='center')

            self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive plot."""
        plt.show()


class StateTrajectoryVisualizer:
    """
    Interactive 3D visualization of hidden or cell state trajectories.
    """

    def __init__(self, pca_dir: Path, state_type: str = 'hidden'):
        """
        Initialize trajectory visualizer.

        Args:
            pca_dir: Path to pca_preprocessed directory
            state_type: 'hidden' or 'cell'
        """
        self.pca_dir = pca_dir
        self.state_type = state_type
        self.state_key = 'h_t_pca' if state_type == 'hidden' else 'c_t_pca'
        self.var_key = 'h_t_explained_variance' if state_type == 'hidden' else 'c_t_explained_variance'

        # Load all PCA files
        self.load_data()

    def load_data(self):
        """Load all PCA preprocessed data."""
        pca_files = sorted(self.pca_dir.glob('epoch_*_pca_states.npz'))

        self.data = {}
        self.epochs = []

        for file in pca_files:
            data = np.load(file)
            epoch = data['epoch'].item()
            self.epochs.append(epoch)

            self.data[epoch] = {
                'states': data[self.state_key],  # (n_periods, timesteps, 3)
                'periods': data['periods'],
                'mse': data['mse_per_period'],
                'explained_variance': data[self.var_key]
            }

        self.epochs.sort()
        self.n_periods = len(self.data[self.epochs[0]]['periods'])

    def create_trajectory_plot(self, epoch_idx: int = 0, period_idx: int = 0) -> go.Figure:
        """
        Create 3D trajectory plot for specific epoch and period.

        Args:
            epoch_idx: Index of epoch
            period_idx: Index of period

        Returns:
            Plotly figure object
        """
        epoch = self.epochs[epoch_idx]
        data = self.data[epoch]

        # Get trajectory for this period
        trajectory = data['states'][period_idx]  # (timesteps, 3)
        period = data['periods'][period_idx]
        mse = data['mse'][period_idx]
        explained_var = data['explained_variance'][period_idx]

        # Create time colors
        timesteps = len(trajectory)
        colors = np.linspace(0, 1, timesteps)

        # Create 3D scatter plot with line
        fig = go.Figure()

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            line=dict(
                color=colors,
                colorscale='Viridis',
                width=3,
                showscale=True,
                colorbar=dict(
                    title='Time',
                    thickness=15,
                    len=0.5
                )
            ),
            marker=dict(
                size=2,
                color=colors,
                colorscale='Viridis',
                showscale=False
            ),
            name='Trajectory',
            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<br>'
        ))

        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start',
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='End',
            showlegend=True
        ))

        # Update layout
        total_var = explained_var.sum()
        fig.update_layout(
            title=f'{self.state_type.capitalize()} States - Epoch {epoch}, Period {period:.3f}s<br>' +
                  f'MSE: {mse:.6f} | Variance Explained: {total_var:.3f}',
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.1%})',
                yaxis_title=f'PC2 ({explained_var[1]:.1%})',
                zaxis_title=f'PC3 ({explained_var[2]:.1%})',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            showlegend=True
        )

        return fig

    def create_interactive_plot(self):
        """
        Create interactive plot with epoch and period sliders.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure
        fig = self.create_trajectory_plot(0, 0)

        # Get initial data for slider ranges
        epoch = self.epochs[0]
        data = self.data[epoch]

        # Create slider steps for epochs
        epoch_steps = []
        for i, epoch in enumerate(self.epochs):
            epoch_steps.append({
                'args': [None, {'frame': {'duration': 0},
                               'mode': 'immediate',
                               'transition': {'duration': 0}}],
                'label': str(epoch),
                'method': 'animate'
            })

        # Create slider steps for periods
        period_steps = []
        for i, period in enumerate(data['periods']):
            period_steps.append({
                'args': [None, {'frame': {'duration': 0},
                               'mode': 'immediate',
                               'transition': {'duration': 0}}],
                'label': f'{period:.3f}',
                'method': 'animate'
            })

        # Add sliders
        sliders = [
            {
                'active': 0,
                'steps': epoch_steps,
                'y': 0,
                'len': 0.4,
                'x': 0.1,
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Epoch: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0}
            },
            {
                'active': 0,
                'steps': period_steps,
                'y': 0,
                'len': 0.4,
                'x': 0.55,
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Period: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0}
            }
        ]

        fig.update_layout(
            sliders=sliders,
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500},
                                      'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0},
                                        'mode': 'immediate',
                                        'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        return fig

    def save_html(self, output_path: Path):
        """Save interactive plot as HTML."""
        fig = self.create_interactive_plot()
        fig.write_html(str(output_path))
        print(f"Saved interactive plot to {output_path}")


def main():
    """
    Main function to create visualizations.
    """
    parser = argparse.ArgumentParser(description='Visualize LSTM dynamics')
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--plot_type', type=str,
                       choices=['error', 'hidden', 'cell', 'all'],
                       default='all',
                       help='Type of plot to create')
    parser.add_argument('--save_html', action='store_true',
                       help='Save interactive plots as HTML')

    args = parser.parse_args()

    # Setup paths
    experiment_path = Path(args.experiment_path)
    state_analysis_dir = experiment_path / 'state_analysis'
    pca_dir = state_analysis_dir / 'pca_preprocessed'
    viz_dir = state_analysis_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # Create plots based on type
    if args.plot_type in ['error', 'all']:
        print("Creating error progression plot...")
        accuracy_file = state_analysis_dir / 'accuracy_progression.json'
        if accuracy_file.exists():
            plotter = ErrorProgressionPlotter(accuracy_file)
            plotter.show()
        else:
            print(f"Accuracy file not found: {accuracy_file}")

    if args.plot_type in ['hidden', 'all']:
        print("Creating hidden state trajectory plot...")
        if pca_dir.exists():
            visualizer = StateTrajectoryVisualizer(pca_dir, 'hidden')
            if args.save_html:
                output_file = viz_dir / 'hidden_states_interactive.html'
                visualizer.save_html(output_file)
            else:
                # Create matplotlib version for display
                for epoch_idx in range(len(visualizer.epochs)):
                    for period_idx in range(visualizer.n_periods):
                        fig = visualizer.create_trajectory_plot(epoch_idx, period_idx)
                        fig.show()
                        break  # Show just first period as example
                    break  # Show just first epoch as example
        else:
            print(f"PCA directory not found: {pca_dir}")

    if args.plot_type in ['cell', 'all']:
        print("Creating cell state trajectory plot...")
        if pca_dir.exists():
            visualizer = StateTrajectoryVisualizer(pca_dir, 'cell')
            if args.save_html:
                output_file = viz_dir / 'cell_states_interactive.html'
                visualizer.save_html(output_file)
            else:
                # Create matplotlib version for display
                for epoch_idx in range(len(visualizer.epochs)):
                    for period_idx in range(visualizer.n_periods):
                        fig = visualizer.create_trajectory_plot(epoch_idx, period_idx)
                        fig.show()
                        break  # Show just first period as example
                    break  # Show just first epoch as example
        else:
            print(f"PCA directory not found: {pca_dir}")

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
