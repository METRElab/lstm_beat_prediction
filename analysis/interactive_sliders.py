"""
Create fully interactive plots with dual sliders using matplotlib.
This version creates plots that work directly in Python without needing HTML export.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class ErrorProgressionSliderPlot:
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

        # Store text annotations
        self.annotations = []

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
            for ann in self.annotations:
                ann.remove()
            self.annotations.clear()

            # Add value labels
            for period, mse in zip(periods, mse_values):
                ann = self.ax.annotate(f'{mse:.4f}',
                                      xy=(period, mse),
                                      xytext=(0, 5),
                                      textcoords='offset points',
                                      fontsize=8,
                                      ha='center')
                self.annotations.append(ann)

            self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive plot."""
        plt.show()


# class DualSlider3DPlot:
#     """
#     Interactive 3D plot with epoch and period sliders for state trajectories.
#     """
#
#     def __init__(self, pca_dir: Path, state_type: str = 'hidden'):
#         """
#         Initialize dual slider 3D plot.
#
#         Args:
#             pca_dir: Path to pca_preprocessed directory
#             state_type: 'hidden' or 'cell'
#         """
#         self.pca_dir = pca_dir
#         self.state_type = state_type
#         self.state_key = 'h_t_pca' if state_type == 'hidden' else 'c_t_pca'
#         self.var_key = 'h_t_explained_variance' if state_type == 'hidden' else 'c_t_explained_variance'
#
#         # Load data
#         self.load_data()
#
#         # Load experiment config to get task parameters
#         self.load_experiment_config()
#
#         # Load original state files with outputs
#         self.load_outputs()
#
#         self.load_phase_infos()  # ADD THIS LINE
#
#         # Setup figure
#         self.setup_plot()
#
#     def load_phase_infos(self):
#         """Load phase infos from state files if available."""
#         self.phase_infos = {}
#         state_analysis_dir = self.pca_dir.parent
#
#         for epoch in self.epochs:
#             state_file = state_analysis_dir / f'epoch_{epoch}_states.npz'
#             if state_file.exists():
#                 data = np.load(state_file, allow_pickle=True)
#
#                 # Check for phase info arrays
#                 phase_keys = [k for k in data.files if k.startswith('phase_')]
#
#                 # Skip if no phase info or if it's a legacy experiment
#                 if not phase_keys:
#                     continue
#
#                 # Check if this is actually array data (not scalar)
#                 first_key = phase_keys[0]
#                 try:
#                     test_array = data[first_key]
#                     # Check if it's actually an array with multiple elements
#                     if test_array.ndim == 0 or len(test_array) == 0:
#                         continue
#                 except (IndexError, TypeError):
#                     continue
#
#                 self.phase_infos[epoch] = {}
#                 n_periods = len(self.periods)
#
#                 for period_idx in range(n_periods):
#                     phase_info = {}
#                     for key in phase_keys:
#                         clean_key = key.replace('phase_', '')
#                         try:
#                             arr = data[key]
#                             if arr.ndim > 0 and period_idx < len(arr):
#                                 phase_info[clean_key] = int(arr[period_idx])
#                             else:
#                                 phase_info[clean_key] = 0
#                         except (IndexError, TypeError):
#                             phase_info[clean_key] = 0
#
#                     self.phase_infos[epoch][period_idx] = phase_info
#
#     def load_experiment_config(self):
#         """Load experiment configuration for task parameters."""
#         # Go up from pca_preprocessed to find config
#         experiment_path = self.pca_dir.parent.parent
#         config_path = experiment_path / 'config.json'
#
#         if config_path.exists():
#             with open(config_path, 'r') as f:
#                 self.config = json.load(f)
#         else:
#             print(f"Warning: Config file not found at {config_path}")
#             self.config = None
#
#     def load_outputs(self):
#         """Load network outputs from original state files."""
#         self.outputs = {}
#         state_analysis_dir = self.pca_dir.parent
#
#         for epoch in self.epochs:
#             state_file = state_analysis_dir / f'epoch_{epoch}_states.npz'
#             if state_file.exists():
#                 data = np.load(state_file)
#                 if 'network_outputs' in data:
#                     self.outputs[epoch] = data['network_outputs']  # (n_periods, timesteps)
#                 else:
#                     print(f"Warning: No outputs found in {state_file}")
#                     self.outputs[epoch] = None
#             else:
#                 self.outputs[epoch] = None
#
#     def calculate_beat_timesteps(self, period: float) -> list:
#         """
#         Calculate timestep indices where beats occur.
#
#         Args:
#             period: Beat period in seconds
#
#         Returns:
#             List of timestep indices for beat onsets
#         """
#         if self.config is None:
#             return []
#
#         task = self.config['task']
#         n_pulses = task['n_pulses']
#         dt = task['dt']
#         # Use fixed ITI from analysis config (min_iti = mean_iti)
#         iti = 2
#
#         beat_timesteps = []
#         for i in range(n_pulses):
#             beat_time = iti/2 + i * period
#             beat_timestep = int(beat_time / dt)
#             beat_timesteps.append(beat_timestep)
#
#         return beat_timesteps
#
#     def load_data(self):
#         """Load all PCA preprocessed data."""
#         pca_files = sorted(self.pca_dir.glob('epoch_*_pca_states.npz'))
#
#         self.data = {}
#         self.epochs = []
#
#         for file in pca_files:
#             data = np.load(file)
#             epoch = data['epoch'].item()
#             self.epochs.append(epoch)
#
#             self.data[epoch] = {
#                 'states': data[self.state_key],
#                 'periods': data['periods'],
#                 'mse': data['mse_per_period'],
#                 'explained_variance': data[self.var_key]
#             }
#
#         self.epochs.sort()
#         self.periods = self.data[self.epochs[0]]['periods']
#         self.n_epochs = len(self.epochs)
#         self.n_periods = len(self.periods)
#
#     def setup_plot(self):
#         """Setup the matplotlib figure with output subplot and 3D axis with sliders."""
#         self.fig = plt.figure(figsize=(12, 12))
#
#         # Create output axis at the top
#         self.ax_output = self.fig.add_axes([0.1, 0.85, 0.8, 0.08])
#
#         # Create 3D axis below
#         self.ax = self.fig.add_axes([0.1, 0.18, 0.8, 0.65], projection='3d')
#
#         # Create sliders at the bottom
#         ax_epoch = plt.axes([0.15, 0.10, 0.7, 0.03])
#         ax_period = plt.axes([0.15, 0.05, 0.7, 0.03])
#
#         self.epoch_slider = Slider(
#             ax_epoch, 'Epoch',
#             0, self.n_epochs - 1,
#             valinit=0, valstep=1,
#             valfmt='%d'
#         )
#
#         self.period_slider = Slider(
#             ax_period, 'Period',
#             0, self.n_periods - 1,
#             valinit=0, valstep=1,
#             valfmt='%d'
#         )
#
#         # Connect sliders to update function
#         self.epoch_slider.on_changed(self.update_plot)
#         self.period_slider.on_changed(self.update_plot)
#
#         # Initialize plot
#         self.trajectory_line = None
#         self.start_point = None
#         self.end_point = None
#         self.scatter = None
#
#         # Initial plot
#         self.update_plot(None)
#
#     def update_plot(self, val):
#         """Update both output and 3D plots based on slider values."""
#         self.ax.clear()
#         self.ax_output.clear()
#
#         epoch_idx = int(self.epoch_slider.val)
#         period_idx = int(self.period_slider.val)
#
#         epoch = self.epochs[epoch_idx]
#         period = self.periods[period_idx]
#         data = self.data[epoch]
#
#         trajectory = data['states'][period_idx]
#         mse = data['mse'][period_idx]
#         explained_var = data['explained_variance'][period_idx]
#
#         # Get phase info if available
#         phase_info = None
#         if hasattr(self, 'phase_infos') and self.phase_infos:
#             if epoch in self.phase_infos:
#                 phase_info = self.phase_infos[epoch].get(period_idx, None)
#
#         # Calculate beat timesteps
#         beat_timesteps = self.calculate_beat_timesteps(period)
#
#         # Plot output if available
#         if epoch in self.outputs and self.outputs[epoch] is not None:
#             output = self.outputs[epoch][period_idx]
#             timesteps_count = len(output)
#
#             if self.config:
#                 dt = self.config['task']['dt']
#                 time_axis = np.arange(timesteps_count) * dt
#                 self.ax_output.plot(time_axis, output, 'b-', linewidth=1.5, label='Network Output')
#                 self.ax_output.set_xlabel('Time (s)')
#             else:
#                 time_axis = np.arange(timesteps_count)
#                 self.ax_output.plot(output, 'b-', linewidth=1.5, label='Network Output')
#                 self.ax_output.set_xlabel('Timesteps')
#
#             # Add phase shading if available
#             if phase_info is not None and self.config:
#                 dt = self.config['task']['dt']
#                 att_end = phase_info.get('attention_end_idx', 0) * dt
#                 skip_end = phase_info.get('skipped_sync_end_idx', att_end) * dt
#                 sync_end = phase_info.get('sync_end_idx', skip_end) * dt
#                 cont_end = phase_info.get('continuation_end_idx', timesteps_count) * dt
#                 max_time = time_axis[-1]
#
#                 # Add phase shading
#                 if att_end > 0:
#                     self.ax_output.axvspan(0, att_end, color='blue', alpha=0.1, label='Attention')
#                 if skip_end > att_end:
#                     self.ax_output.axvspan(att_end, skip_end, color='red', alpha=0.1, label='Skipped')
#                 if sync_end > skip_end:
#                     self.ax_output.axvspan(skip_end, sync_end, color='green', alpha=0.1, label='Sync')
#                 if cont_end > sync_end:
#                     self.ax_output.axvspan(sync_end, cont_end, color='orange', alpha=0.1, label='Continuation')
#                 if cont_end < max_time:
#                     self.ax_output.axvspan(cont_end, max_time, color='gray', alpha=0.1, label='Tail')
#
#             # Add beat markers
#             if beat_timesteps:
#                 valid_beats = [t for t in beat_timesteps if t < len(output)]
#                 if valid_beats:
#                     if self.config:
#                         beat_times = [t * dt for t in valid_beats]
#                         beat_values = output[valid_beats]
#                         self.ax_output.scatter(
#                             beat_times, beat_values,
#                             c='red', s=100, marker='*',
#                             edgecolors='darkred', linewidths=2,
#                             label=f'Beats (n={len(valid_beats)})', zorder=5
#                         )
#
#             self.ax_output.set_ylabel('Output Value')
#             self.ax_output.set_title(f'Network Output - Epoch {epoch}, Period {period:.3f}s, MSE: {mse:.6f}')
#             self.ax_output.grid(True, alpha=0.3)
#             self.ax_output.legend(loc='upper right', fontsize=8)
#
#     def show(self):
#         """Display the interactive plot."""
#         plt.show()


class DualSlider3DPlot:
    """
    Interactive 3D plot with epoch and period sliders for state trajectories.
    """

    def __init__(self, pca_dir: Path, state_type: str = 'hidden'):
        """
        Initialize dual slider 3D plot.

        Args:
            pca_dir: Path to pca_preprocessed directory
            state_type: 'hidden' or 'cell'
        """
        self.pca_dir = pca_dir
        self.state_type = state_type
        self.state_key = 'h_t_pca' if state_type == 'hidden' else 'c_t_pca'
        self.var_key = 'h_t_explained_variance' if state_type == 'hidden' else 'c_t_explained_variance'

        # Load data
        self.load_data()

        # Load experiment config to get task parameters
        self.load_experiment_config()

        # Load original state files with outputs
        self.load_outputs()

        # Load inputs and targets
        self.load_inputs_and_targets()

        # Load phase infos (for sync_continuation tasks)
        self.load_phase_infos()

        # Setup figure
        self.setup_plot()

    def load_experiment_config(self):
        """Load experiment configuration for task parameters."""
        experiment_path = self.pca_dir.parent.parent
        config_path = experiment_path / 'config.json'

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print(f"Warning: Config file not found at {config_path}")
            self.config = None

    def load_outputs(self):
        """Load network outputs from original state files."""
        self.outputs = {}
        state_analysis_dir = self.pca_dir.parent

        for epoch in self.epochs:
            state_file = state_analysis_dir / f'epoch_{epoch}_states.npz'
            if state_file.exists():
                data = np.load(state_file)
                if 'network_outputs' in data:
                    self.outputs[epoch] = data['network_outputs']
                else:
                    self.outputs[epoch] = None
            else:
                self.outputs[epoch] = None

    def load_inputs_and_targets(self):
        """Load network inputs and targets from state files."""
        self.inputs = {}
        self.targets = {}
        state_analysis_dir = self.pca_dir.parent

        for epoch in self.epochs:
            state_file = state_analysis_dir / f'epoch_{epoch}_states.npz'
            if state_file.exists():
                data = np.load(state_file)
                if 'network_inputs' in data:
                    self.inputs[epoch] = data['network_inputs']  # (n_periods, timesteps)
                else:
                    self.inputs[epoch] = None

                if 'target_outputs' in data:
                    self.targets[epoch] = data['target_outputs']  # (n_periods, timesteps)
                else:
                    self.targets[epoch] = None
            else:
                self.inputs[epoch] = None
                self.targets[epoch] = None

    def load_phase_infos(self):
        """Load phase infos from state files if available."""
        self.phase_infos = {}
        state_analysis_dir = self.pca_dir.parent

        for epoch in self.epochs:
            state_file = state_analysis_dir / f'epoch_{epoch}_states.npz'
            if state_file.exists():
                data = np.load(state_file, allow_pickle=True)

                # Check for phase info arrays
                phase_keys = [k for k in data.files if k.startswith('phase_')]

                # Skip if no phase info
                if not phase_keys:
                    continue

                # Check if this is actually array data with proper dimensions
                first_key = phase_keys[0]
                try:
                    test_array = data[first_key]
                    # Check if it's actually an array with multiple elements
                    if not hasattr(test_array, 'ndim') or test_array.ndim == 0 or len(test_array) == 0:
                        continue
                except (IndexError, TypeError, AttributeError):
                    continue

                self.phase_infos[epoch] = {}
                n_periods = len(self.periods)

                for period_idx in range(n_periods):
                    phase_info = {}
                    for key in phase_keys:
                        clean_key = key.replace('phase_', '')
                        try:
                            arr = data[key]
                            if hasattr(arr, 'ndim') and arr.ndim > 0 and period_idx < len(arr):
                                phase_info[clean_key] = int(arr[period_idx])
                            else:
                                phase_info[clean_key] = 0
                        except (IndexError, TypeError, AttributeError):
                            phase_info[clean_key] = 0

                    self.phase_infos[epoch][period_idx] = phase_info

    def calculate_beat_timesteps(self, period: float) -> list:
        """
        Calculate timestep indices where beats occur.
        """
        if self.config is None:
            return []

        task = self.config['task']
        task_type = task.get('task_type', 'legacy')
        dt = task['dt']

        beat_timesteps = []

        if task_type == 'sync_continuation':
            # For sync_continuation, we don't have fixed beat positions
            # They vary per trial, so return empty (beats will come from phase_info)
            return []
        else:
            # Legacy behavior
            n_pulses = task.get('n_pulses', 5)
            iti = 2  # Fixed ITI for analysis

            for i in range(n_pulses):
                beat_time = iti / 2 + i * period
                beat_timestep = int(beat_time / dt)
                beat_timesteps.append(beat_timestep)

        return beat_timesteps

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
                'states': data[self.state_key],
                'periods': data['periods'],
                'mse': data['mse_per_period'],
                'explained_variance': data[self.var_key]
            }

        self.epochs.sort()
        self.periods = self.data[self.epochs[0]]['periods']
        self.n_epochs = len(self.epochs)
        self.n_periods = len(self.periods)

    def setup_plot(self):
        """Setup the matplotlib figure with input, output subplots and 3D axis with sliders."""
        self.fig = plt.figure(figsize=(12, 14))

        # Create input axis at the top
        self.ax_input = self.fig.add_axes([0.1, 0.88, 0.8, 0.06])

        # Create output axis below input
        self.ax_output = self.fig.add_axes([0.1, 0.80, 0.8, 0.06])

        # Create 3D axis below
        self.ax = self.fig.add_axes([0.1, 0.18, 0.8, 0.58], projection='3d')

        # Create sliders at the bottom
        ax_epoch = plt.axes([0.15, 0.10, 0.7, 0.03])
        ax_period = plt.axes([0.15, 0.05, 0.7, 0.03])

        self.epoch_slider = Slider(
            ax_epoch, 'Epoch',
            0, self.n_epochs - 1,
            valinit=0, valstep=1,
            valfmt='%d'
        )

        self.period_slider = Slider(
            ax_period, 'Period',
            0, self.n_periods - 1,
            valinit=0, valstep=1,
            valfmt='%d'
        )

        # Connect sliders to update function
        self.epoch_slider.on_changed(self.update_plot)
        self.period_slider.on_changed(self.update_plot)

        # Initialize plot
        self.trajectory_line = None
        self.start_point = None
        self.end_point = None
        self.scatter = None
        self.colorbar = None

        # Initial plot
        self.update_plot(None)

    def update_plot(self, val):
        """Update input, output, and 3D plots based on slider values."""
        # Clear previous plots
        self.ax.clear()
        self.ax_output.clear()
        self.ax_input.clear()

        # Get current indices
        epoch_idx = int(self.epoch_slider.val)
        period_idx = int(self.period_slider.val)

        # Get data
        epoch = self.epochs[epoch_idx]
        period = self.periods[period_idx]
        data = self.data[epoch]

        # Get trajectory
        trajectory = data['states'][period_idx]  # (timesteps, 3)
        mse = data['mse'][period_idx]
        explained_var = data['explained_variance'][period_idx]

        # Get phase info if available
        phase_info = None
        if hasattr(self, 'phase_infos') and self.phase_infos:
            if epoch in self.phase_infos:
                phase_info = self.phase_infos[epoch].get(period_idx, None)

        # Calculate beat timesteps (for legacy tasks)
        beat_timesteps = self.calculate_beat_timesteps(period)

        # Get time axis
        if self.config:
            dt = self.config['task']['dt']
        else:
            dt = 0.01  # Default

        # ============ PLOT INPUT ============
        if hasattr(self, 'inputs') and epoch in self.inputs and self.inputs[epoch] is not None:
            input_seq = self.inputs[epoch][period_idx]
            timesteps_count = len(input_seq)
            time_axis = np.arange(timesteps_count) * dt

            self.ax_input.plot(time_axis, input_seq, 'b-', linewidth=1.5, label='Input')
            self.ax_input.set_ylabel('Input', fontsize=9)
            self.ax_input.set_xlim(time_axis[0], time_axis[-1])
            self.ax_input.grid(True, alpha=0.3)
            self.ax_input.legend(loc='upper right', fontsize=8)

            # Add phase shading to input
            if phase_info is not None:
                att_end = phase_info.get('attention_end_idx', 0) * dt
                skip_end = phase_info.get('skipped_sync_end_idx', 0) * dt
                sync_end = phase_info.get('sync_end_idx', 0) * dt
                cont_end = phase_info.get('continuation_end_idx', timesteps_count) * dt
                max_time = time_axis[-1]

                if att_end > 0:
                    self.ax_input.axvspan(0, att_end, color='blue', alpha=0.1)
                if skip_end > att_end:
                    self.ax_input.axvspan(att_end, skip_end, color='red', alpha=0.1)
                if sync_end > skip_end:
                    self.ax_input.axvspan(skip_end, sync_end, color='green', alpha=0.1)
                if cont_end > sync_end:
                    self.ax_input.axvspan(sync_end, cont_end, color='orange', alpha=0.1)
                if cont_end < max_time:
                    self.ax_input.axvspan(cont_end, max_time, color='gray', alpha=0.1)

            self.ax_input.set_title(f'Epoch {epoch} | Period {period:.3f}s | MSE: {mse:.6f}', fontsize=11)
        else:
            self.ax_input.text(0.5, 0.5, 'Input data not available\nRe-run analyze_states.py',
                               ha='center', va='center', transform=self.ax_input.transAxes)
            self.ax_input.set_title(f'Epoch {epoch} | Period {period:.3f}s | MSE: {mse:.6f}', fontsize=11)

        # Remove x-axis labels from input plot (shared with output below)
        self.ax_input.set_xticklabels([])

        # ============ PLOT OUTPUT AND TARGET ============
        if epoch in self.outputs and self.outputs[epoch] is not None:
            output = self.outputs[epoch][period_idx]
            timesteps_count = len(output)
            time_axis = np.arange(timesteps_count) * dt

            # Plot target if available
            if hasattr(self, 'targets') and epoch in self.targets and self.targets[epoch] is not None:
                target = self.targets[epoch][period_idx]
                self.ax_output.plot(time_axis, target, 'k--', linewidth=1.5, alpha=0.7, label='Target')

            # Plot network output
            self.ax_output.plot(time_axis, output, 'r-', linewidth=1.5, label='Output')

            self.ax_output.set_xlabel('Time (s)', fontsize=9)
            self.ax_output.set_ylabel('Output', fontsize=9)
            self.ax_output.set_xlim(time_axis[0], time_axis[-1])
            self.ax_output.grid(True, alpha=0.3)
            self.ax_output.legend(loc='upper right', fontsize=8)

            # Add phase shading to output
            if phase_info is not None:
                att_end = phase_info.get('attention_end_idx', 0) * dt
                skip_end = phase_info.get('skipped_sync_end_idx', 0) * dt
                sync_end = phase_info.get('sync_end_idx', 0) * dt
                cont_end = phase_info.get('continuation_end_idx', timesteps_count) * dt
                max_time = time_axis[-1]

                if att_end > 0:
                    self.ax_output.axvspan(0, att_end, color='blue', alpha=0.1)
                if skip_end > att_end:
                    self.ax_output.axvspan(att_end, skip_end, color='red', alpha=0.1)
                if sync_end > skip_end:
                    self.ax_output.axvspan(skip_end, sync_end, color='green', alpha=0.1)
                if cont_end > sync_end:
                    self.ax_output.axvspan(sync_end, cont_end, color='orange', alpha=0.1)
                if cont_end < max_time:
                    self.ax_output.axvspan(cont_end, max_time, color='gray', alpha=0.1)

            # Add beat markers (for legacy tasks)
            if beat_timesteps:
                valid_beats = [t for t in beat_timesteps if t < len(output)]
                if valid_beats:
                    beat_times = [t * dt for t in valid_beats]
                    beat_values = [output[t] for t in valid_beats]
                    self.ax_output.scatter(
                        beat_times, beat_values,
                        c='red', s=100, marker='*',
                        edgecolors='darkred', linewidths=2,
                        zorder=5
                    )
        else:
            self.ax_output.text(0.5, 0.5, 'Output data not available\nRe-run analyze_states.py',
                                ha='center', va='center', transform=self.ax_output.transAxes)

        # ============ PLOT 3D TRAJECTORY ============
        timesteps = len(trajectory)

        # Plot trajectory as scatter with color gradient
        self.scatter = self.ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            c=np.arange(timesteps),
            cmap='viridis',
            s=3,
            alpha=0.6
        )

        # Plot line connecting points
        self.ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            'b-', linewidth=0.5, alpha=0.3
        )

        # Mark start and end
        self.ax.scatter(
            [trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
            c='green', s=100, marker='o', label='Start'
        )
        self.ax.scatter(
            [trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
            c='red', s=100, marker='s', label='End'
        )

        # Add phase boundary markers on 3D plot if available
        if phase_info is not None:
            boundaries = [
                ('attention_end_idx', 'Att→Skip', 'blue'),
                ('skipped_sync_end_idx', 'Skip→Sync', 'red'),
                ('sync_end_idx', 'Sync→Cont', 'green'),
                ('continuation_end_idx', 'Cont→Tail', 'orange'),
            ]

            for key, label, color in boundaries:
                idx = phase_info.get(key, 0)
                if 0 < idx < len(trajectory):
                    self.ax.scatter(
                        [trajectory[idx, 0]], [trajectory[idx, 1]], [trajectory[idx, 2]],
                        c=color, s=80, marker='D', label=label
                    )

        # Add beat markers on 3D trajectory (for legacy tasks)
        if beat_timesteps:
            valid_beats = [t for t in beat_timesteps if t < len(trajectory)]
            if valid_beats:
                for i, beat_idx in enumerate(valid_beats, 1):
                    beat_point = trajectory[beat_idx]
                    self.ax.text(beat_point[0], beat_point[1], beat_point[2], str(i),
                                 color='red', fontsize=12, fontweight='bold',
                                 ha='center', va='center',
                                 bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', alpha=0.8))

        # Set labels
        self.ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        self.ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        self.ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})')

        # Set title
        total_var = explained_var.sum()
        self.ax.set_title(
            f'{self.state_type.capitalize()} States | Variance Explained: {total_var:.1%}',
            fontsize=11
        )

        # Update slider labels
        self.epoch_slider.valtext.set_text(f'{epoch}')
        self.period_slider.valtext.set_text(f'{period:.3f}s')

        # Add legend
        self.ax.legend(loc='upper right', fontsize=8)

        # Add colorbar only once
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(
                self.scatter, ax=self.ax,
                label='Time (steps)',
                shrink=0.5,
                aspect=5
            )

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive plot."""
        plt.show()

class ComparisonPlot:
    """
    Plot showing all trajectories for a given epoch or period.
    """

    def __init__(self, pca_dir: Path, state_type: str = 'hidden'):
        """Initialize comparison plot."""
        self.pca_dir = pca_dir
        self.state_type = state_type
        self.state_key = 'h_t_pca' if state_type == 'hidden' else 'c_t_pca'

        # Load data
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
                'states': data[self.state_key],
                'periods': data['periods'],
                'mse': data['mse_per_period']
            }

        self.epochs.sort()
        self.periods = self.data[self.epochs[0]]['periods']

    def plot_all_periods_for_epoch(self, epoch_idx: int = -1):
        """
        Plot all period trajectories for a specific epoch.

        Args:
            epoch_idx: Index of epoch (-1 for last)
        """
        epoch = self.epochs[epoch_idx]
        data = self.data[epoch]

        fig = plt.figure(figsize=(15, 10))
        n_periods = len(self.periods)

        # Create colormap for periods
        colors = plt.cm.rainbow(np.linspace(0, 1, n_periods))

        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Plot each period's trajectory
        for i, period in enumerate(self.periods):
            trajectory = data['states'][i]

            # Subsample for visualization
            step = max(1, len(trajectory) // 100)
            traj_sub = trajectory[::step]

            ax.plot(
                traj_sub[:, 0],
                traj_sub[:, 1],
                traj_sub[:, 2],
                color=colors[i],
                alpha=0.6,
                linewidth=1,
                label=f'{period:.2f}s'
            )

            # Mark start points
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                c=[colors[i]],
                s=50,
                marker='o'
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{self.state_type.capitalize()} State Trajectories - Epoch {epoch}')
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_epoch_progression_for_period(self, period_idx: int = 0):
        """
        Plot trajectory evolution across epochs for a specific period.

        Args:
            period_idx: Index of period to plot
        """
        period = self.periods[period_idx]

        fig = plt.figure(figsize=(15, 10))

        # Create colormap for epochs
        n_epochs = len(self.epochs)
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_epochs))

        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory for each epoch
        for i, epoch in enumerate(self.epochs):
            trajectory = self.data[epoch]['states'][period_idx]

            # Subsample for visualization
            step = max(1, len(trajectory) // 100)
            traj_sub = trajectory[::step]

            ax.plot(
                traj_sub[:, 0],
                traj_sub[:, 1],
                traj_sub[:, 2],
                color=colors[i],
                alpha=0.6,
                linewidth=1,
                label=f'Epoch {epoch}'
            )

            # Mark start points
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                c=[colors[i]],
                s=50,
                marker='o'
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{self.state_type.capitalize()} State Evolution - Period {period:.3f}s')
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


def main():
    """Main function to create interactive visualizations."""
    parser = argparse.ArgumentParser(description='Interactive LSTM dynamics visualization')
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--plot_type', type=str,
                       choices=['hidden', 'cell', 'error', 'all'],
                       default='all',
                       help='Type of plot to visualize')
    parser.add_argument('--comparison', action='store_true',
                       help='Show comparison plots (for state plots only)')

    args = parser.parse_args()

    # Setup paths
    experiment_path = Path(args.experiment_path)
    state_analysis_dir = experiment_path / 'state_analysis'
    pca_dir = state_analysis_dir / 'pca_preprocessed'
    accuracy_file = state_analysis_dir / 'accuracy_progression.json'

    # Create error progression plot
    if args.plot_type in ['error', 'all']:
        if accuracy_file.exists():
            print("Creating error progression visualization...")
            error_plot = ErrorProgressionSliderPlot(accuracy_file)
            error_plot.show()
        else:
            print(f"Error: Accuracy file not found: {accuracy_file}")

    # Create state visualizations
    if args.plot_type in ['hidden', 'cell', 'all']:
        if not pca_dir.exists():
            print(f"Error: PCA directory not found: {pca_dir}")
            print("Please run preprocess_pca.py first")
            return

    # Create hidden state visualization
    if args.plot_type in ['hidden', 'all']:
        print("Creating hidden state visualization...")
        hidden_plot = DualSlider3DPlot(pca_dir, 'hidden')

        if args.comparison:
            comp_plot = ComparisonPlot(pca_dir, 'hidden')
            comp_plot.plot_all_periods_for_epoch(-1)  # Last epoch
            comp_plot.plot_epoch_progression_for_period(0)  # First period

        hidden_plot.show()

    # Create cell state visualization
    if args.plot_type in ['cell', 'all']:
        print("Creating cell state visualization...")
        cell_plot = DualSlider3DPlot(pca_dir, 'cell')

        if args.comparison:
            comp_plot = ComparisonPlot(pca_dir, 'cell')
            comp_plot.plot_all_periods_for_epoch(-1)  # Last epoch
            comp_plot.plot_epoch_progression_for_period(0)  # First period

        cell_plot.show()


if __name__ == '__main__':
    main()
