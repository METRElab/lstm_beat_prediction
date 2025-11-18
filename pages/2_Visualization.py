"""
Main visualization page for model interaction and analysis.
"""

import streamlit as st
from pathlib import Path
import numpy as np
import json
import pickle
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import sys
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_runner import ModelRunner
from utils.input_generator import generate_noisy_pulse_times, create_input_tensor_from_times
from utils.viz_utils import plot_input_output, create_3d_trajectory

st.set_page_config(
    page_title="Model Visualization",
    page_icon="📊",
    layout="wide"
)

# Check if experiment is selected
if 'experiment_path' not in st.session_state or st.session_state.experiment_path is None:
    st.error("❌ No experiment selected!")
    st.info("Please go to **Experiment Selection** page first")
    st.stop()

# Page title
st.title("📊 Interactive Model Visualization")
st.markdown(f"**Selected Experiment:** {st.session_state.selected_experiment}")
st.markdown(f"**Noise Levels:** Phase = {st.session_state.phase_noise:.3f}, Jitter = {st.session_state.jitter_noise:.3f}")

# Initialize model runner
@st.cache_resource
def load_model_runner(experiment_path):
    return ModelRunner(Path(experiment_path))

model_runner = load_model_runner(st.session_state.experiment_path)

# Sidebar for model controls
st.sidebar.title("⚙️ Model Controls")

# Get available epochs
available_epochs = model_runner.get_available_epochs()
if not available_epochs:
    st.error("No checkpoints found in experiment")
    st.stop()

# Epoch selection using index
st.sidebar.markdown("### Epoch Selection")

# Use index-based selection
epoch_index = st.sidebar.slider(
    "Select Epoch (by index):",
    min_value=0,
    max_value=len(available_epochs) - 1,
    value=len(available_epochs) - 1,  # Default to last epoch
    step=1,
    format="Epoch %d"
)

# Get actual epoch number from index
selected_epoch = available_epochs[epoch_index]

# Display the actual epoch number
st.sidebar.info(f"Selected Epoch: **{selected_epoch}** (index {epoch_index + 1} of {len(available_epochs)})")

# Show available epochs
with st.sidebar.expander("Available Epochs"):
    for idx, epoch in enumerate(available_epochs):
        if idx == epoch_index:
            st.markdown(f"**→ {epoch} (selected)**")
        else:
            st.markdown(f"  {epoch}")

# Main content area
tab1, tab2 = st.tabs(["🎹 Model Interface", "🔧 Advanced"])

with tab1:
    # Compact input configuration section
    st.markdown("### Input Configuration")

    # All inputs in one row
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        first_beat_time = st.number_input(
            "First beat time (s)",
            min_value=0.0,
            max_value=5.0,
            value=0.2,
            step=0.05
        )

    with col2:
        tempo = st.number_input(
            "Tempo (s)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.05
        )

    with col3:
        n_pulses = st.number_input(
            "# Pulses",
            min_value=3,
            max_value=10,
            value=5,
            step=1
        )

    with col4:
        sequence_length = st.number_input(
            "Length (s)",
            min_value=1.0,
            max_value=15.0,
            value=7.0,
            step=0.5
        )

    with col5:
        phase_noise_input = st.number_input(
            "Phase σ",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.005,
            format="%.3f"
        )

    with col6:
        jitter_noise_input = st.number_input(
            "Jitter σ",
            min_value=0.0,
            max_value=0.1,
            value=0.005,
            step=0.005,
            format="%.3f"
        )

    with col7:
        generate_btn = st.button("🎲 Generate", type="secondary", use_container_width=True)

    # Generate or display pulse times
    if generate_btn or 'pulse_times' not in st.session_state:
        pulse_times = generate_noisy_pulse_times(
            first_beat_time=first_beat_time,
            tempo=tempo,
            n_pulses=n_pulses,
            phase_noise=phase_noise_input,
            jitter=jitter_noise_input
        )
        st.session_state.pulse_times = pulse_times

    # Pulse times editor in a more compact format
    col1, col2 = st.columns([4, 1])

    with col1:
        # Convert to string for editing
        pulse_times_str = ', '.join([f"{t:.3f}" for t in st.session_state.pulse_times])

        edited_times_str = st.text_input(
            "Pulse times (seconds, comma-separated):",
            value=pulse_times_str,
            help="Edit directly or use Generate button"
        )

        # Parse edited times
        try:
            edited_times = [float(t.strip()) for t in edited_times_str.split(',') if t.strip()]
            st.session_state.pulse_times = edited_times
        except ValueError as e:
            st.error(f"Invalid format: {e}")

    with col2:
        if st.button("🚀 Run Model", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                # Load checkpoint
                model_runner.load_checkpoint(selected_epoch)

                # Generate input from pulse times
                dt = model_runner.config['task']['dt']
                pulse_width = model_runner.config['task']['pulse_width']
                pulse_height = model_runner.config['task']['pulse_height']

                input_tensor = create_input_tensor_from_times(
                    pulse_times=st.session_state.pulse_times,
                    pulse_width=pulse_width,
                    pulse_height=pulse_height,
                    sequence_length=sequence_length,
                    dt=dt
                )

                # Run model
                output, hidden_states, cell_states = model_runner.run_with_states(input_tensor)

                # Load PCA models
                pca_h, pca_c = model_runner.load_pca_models(selected_epoch)

                # Transform states
                hidden_pca = model_runner.transform_states(hidden_states, pca_h)
                cell_pca = model_runner.transform_states(cell_states, pca_c)

                # Store results
                st.session_state.results = {
                    'input': input_tensor,
                    'output': output,
                    'hidden_states': hidden_states,
                    'cell_states': cell_states,
                    'hidden_pca': hidden_pca,
                    'cell_pca': cell_pca,
                    'pulse_times': st.session_state.pulse_times,
                    'dt': dt,
                    'epoch': selected_epoch
                }

                st.success("✅ Complete!")
                st.rerun()

    # Results section (only show if results exist)
    if 'results' in st.session_state:
        st.markdown("---")
        st.markdown("### Results")

        results = st.session_state.results

        # Input/Output plot
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### Input/Output Signals")
        with col2:
            st.info(f"Epoch {results['epoch']} | {len(results['pulse_times'])} pulses")

        fig = plot_input_output(
            input_seq=results['input'],
            output_seq=results['output'],
            pulse_times=results['pulse_times'],
            dt=results['dt']
        )
        st.pyplot(fig)

        # 3D State visualization
        st.markdown("#### 3D State Trajectory")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            state_type = st.radio(
                "State Type:",
                ["Hidden", "Cell"],
                horizontal=True
            )

        with col2:
            show_beats = st.checkbox("Show Beats", value=True)

        with col3:
            # Trajectory stats in same row
            if state_type == "Hidden":
                states_to_plot = results['hidden_pca']
            else:
                states_to_plot = results['cell_pca']

            # traj_length = np.sum(np.linalg.norm(np.diff(states_to_plot, axis=0), axis=1))
            # st.metric("Trajectory Length", f"{traj_length:.2f}")

        # Create 3D plot
        title = f"{state_type} States - Epoch {results['epoch']}"
        fig_3d = create_3d_trajectory(
            states_pca=states_to_plot,
            pulse_times=results['pulse_times'] if show_beats else None,
            dt=results['dt'],
            title=title,
            input_seq=results['input'],
            output_seq=results['output']
        )

        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("👆 Configure input and click 'Run Model' to see results")

with tab2:
    st.markdown("### Advanced Settings")

    if 'results' in st.session_state:
        st.markdown("#### Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Create data for download
            export_data = {
                'pulse_times': st.session_state.pulse_times,
                'input': st.session_state.results['input'].tolist(),
                'output': st.session_state.results['output'].tolist(),
                'hidden_pca': st.session_state.results['hidden_pca'].tolist(),
                'cell_pca': st.session_state.results['cell_pca'].tolist(),
                'epoch': st.session_state.results['epoch'],
                'dt': st.session_state.results['dt']
            }

            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"results_epoch_{selected_epoch}.json",
                mime="application/json"
            )

        with col2:
            st.info("Data includes input, output, and PCA-transformed states")

    # Model info
    st.markdown("#### Model Information")
    with st.expander("View Configuration"):
        st.json(model_runner.config)

    # Debug info
    if st.checkbox("Show Debug Info"):
        st.markdown("#### Session State")
        st.json({k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v
                 for k, v in st.session_state.items()})

# Footer
st.markdown("---")
st.caption("Use the sidebar to change epochs, and the tabs to configure inputs and view results.")
