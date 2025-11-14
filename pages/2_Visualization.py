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
tab1, tab2, tab3 = st.tabs(["🎹 Input Configuration", "📈 Results", "🔧 Advanced"])

with tab1:
    st.markdown("### Configure Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        first_beat_time = st.number_input(
            "First beat time (seconds):",
            min_value=0.0,
            max_value=5.0,
            value=0.2,
            step=0.05,
            help="First beat time of the trial"
        )

        tempo = st.number_input(
            "Tempo (seconds):",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help="Base period between beats"
        )

        n_pulses = st.number_input(
            "Number of Pulses:",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Number of beats to generate"
        )

    with col2:
        sequence_length = st.number_input(
            "Sequence Length (seconds):",
            min_value=1.0,
            max_value=15.0,
            value=7.0,
            step=0.5,
            help="Total duration of the sequence"
        )

        phase_noise_input = st.number_input(
            "Phase Noise STD:",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.005,
            format="%.3f",
            help="Standard deviation of phase noise"
        )

    with col3:
        jitter_noise_input = st.number_input(
            "Jitter Noise STD:",
            min_value=0.0,
            max_value=0.1,
            value=0.005,
            step=0.005,
            format="%.3f",
            help="Standard deviation of jitter noise"
        )

        st.markdown("")  # Spacer
        generate_btn = st.button("🎲 Generate Pulse Times", type="primary", use_container_width=True)

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

    # Display and allow editing of pulse times
    st.markdown("### Pulse Times (seconds)")
    st.info("You can edit these values directly. The model will use whatever you specify.")

    # Convert to string for editing
    pulse_times_str = ', '.join([f"{t:.3f}" for t in st.session_state.pulse_times])

    edited_times_str = st.text_area(
        "Edit pulse times (comma-separated):",
        value=pulse_times_str,
        height=100,
        help="Enter comma-separated values in seconds"
    )

    # Parse edited times
    try:
        edited_times = [float(t.strip()) for t in edited_times_str.split(',') if t.strip()]
        st.session_state.pulse_times = edited_times
        st.success(f"✅ {len(edited_times)} pulse times ready")
    except ValueError as e:
        st.error(f"Invalid format: {e}")

    # Display parsed times
    with st.expander("View Parsed Times"):
        for i, t in enumerate(st.session_state.pulse_times):
            st.text(f"Pulse {i+1}: {t:.3f} seconds")

    # Run model button
    st.markdown("### Run Model")
    run_col1, run_col2 = st.columns([3, 1])

    with run_col1:
        if st.button("🚀 Run Model", type="primary", use_container_width=True):
            with st.spinner("Loading model and running inference..."):
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

                st.success("✅ Model execution complete!")
                st.rerun()

with tab2:
    st.markdown("### Model Results")

    if 'results' not in st.session_state:
        st.info("👈 Configure input and run the model first")
    else:
        results = st.session_state.results

        # Input/Output plot
        st.markdown("#### Input and Output Signals")

        fig = plot_input_output(
            input_seq=results['input'],
            output_seq=results['output'],
            pulse_times=results['pulse_times'],
            dt=results['dt']
        )
        st.pyplot(fig)

        # 3D State visualization
        st.markdown("#### 3D State Trajectory")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            state_type = st.radio(
                "State Type:",
                ["Hidden States", "Cell States"],
                horizontal=True
            )

        with col2:
            show_beats = st.checkbox("Show Beat Markers", value=True)

        with col3:
            st.info(f"Epoch {results['epoch']}")

        # Select which states to show
        if state_type == "Hidden States":
            states_to_plot = results['hidden_pca']
            title = f"Hidden States - Epoch {results['epoch']}"
        else:
            states_to_plot = results['cell_pca']
            title = f"Cell States - Epoch {results['epoch']}"

        # Create 3D plot
        fig_3d = create_3d_trajectory(
            states_pca=states_to_plot,
            pulse_times=results['pulse_times'] if show_beats else None,
            dt=results['dt'],
            title=title,
            input_seq=results['input'],
            output_seq=results['output']
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # Statistics
        with st.expander("📊 Trajectory Statistics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Trajectory Length", f"{np.sum(np.linalg.norm(np.diff(states_to_plot, axis=0), axis=1)):.3f}")

            with col2:
                st.metric("Mean Speed", f"{np.mean(np.linalg.norm(np.diff(states_to_plot, axis=0), axis=1)):.3f}")

            with col3:
                st.metric("Max Distance from Origin", f"{np.max(np.linalg.norm(states_to_plot, axis=1)):.3f}")

with tab3:
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
