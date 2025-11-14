"""
Experiment selection page for choosing models based on noise levels.
"""

import streamlit as st
from pathlib import Path
import re
from typing import Dict, Tuple, List
import json

st.set_page_config(
    page_title="Experiment Selection",
    page_icon="📁",
    layout="wide"
)


def parse_experiment_name(folder_name: str) -> Tuple[float, float]:
    """Parse phase and jitter noise levels from experiment folder name."""
    match = re.search(r'p([\d.]+)_j([\d.]+)', folder_name)
    if match:
        phase = float(match.group(1))
        jitter = float(match.group(2))
        return phase, jitter
    return None, None


def scan_experiments(base_dir: Path) -> Dict:
    """
    Scan directory for experiments and extract noise levels.

    Returns:
        Dictionary with experiment info organized by noise levels
    """
    experiments = {}
    phase_levels = set()
    jitter_levels = set()

    if not base_dir.exists():
        return {'experiments': {}, 'phase_levels': [], 'jitter_levels': []}

    # Find all experiment directories
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and '_p' in exp_dir.name and '_j' in exp_dir.name:
            phase, jitter = parse_experiment_name(exp_dir.name)
            if phase is not None and jitter is not None:
                experiments[(phase, jitter)] = exp_dir
                phase_levels.add(phase)
                jitter_levels.add(jitter)

    return {
        'experiments': experiments,
        'phase_levels': sorted(list(phase_levels)),
        'jitter_levels': sorted(list(jitter_levels))
    }


def check_experiment_validity(exp_path: Path) -> Dict:
    """Check if experiment has required files."""
    checks = {
        'config': (exp_path / 'config.json').exists(),
        'checkpoints': (exp_path / 'checkpoints').exists(),
        'pca_models': (exp_path / 'pca_models').exists()
    }

    # Count checkpoints
    if checks['checkpoints']:
        checkpoint_files = list((exp_path / 'checkpoints').glob('checkpoint_epoch_*.pth'))
        checks['n_checkpoints'] = len(checkpoint_files)
    else:
        checks['n_checkpoints'] = 0

    # Count PCA models
    if checks['pca_models']:
        pca_files = list((exp_path / 'pca_models').glob('epoch_*_pca_h.pkl'))
        checks['n_pca_models'] = len(pca_files)
    else:
        checks['n_pca_models'] = 0

    checks['is_valid'] = all([checks['config'], checks['checkpoints'], checks['pca_models']])

    return checks


# Page title
st.title("📁 Experiment Selection")
st.markdown("Select an experiment based on phase and jitter noise levels")

# Initialize session state
if 'base_dir' not in st.session_state:
    st.session_state.base_dir = 'experiments/gaussian_noise'

# Directory input
col1, col2 = st.columns([3, 1])
with col1:
    base_dir_input = st.text_input(
        "Base directory containing experiments:",
        value=st.session_state.base_dir,
        help="Path to directory containing experiment folders named like '20251109_160331_p0.005_j0.01'"
    )

with col2:
    if st.button("🔍 Scan Directory", use_container_width=True):
        st.session_state.base_dir = base_dir_input
        st.rerun()

# Scan experiments
base_dir = Path(st.session_state.base_dir)
scan_results = scan_experiments(base_dir)

if not scan_results['experiments']:
    st.error(f"No valid experiments found in: {base_dir}")
    st.info("Expected folder names like: '20251109_160331_p0.005_j0.01'")
    st.stop()

# Display found experiments
st.success(f"Found {len(scan_results['experiments'])} experiments")

# Create selection interface
st.markdown("### Select Noise Levels")

col1, col2 = st.columns(2)

with col1:
    selected_phase = st.selectbox(
        "Phase Noise Level:",
        options=scan_results['phase_levels'],
        format_func=lambda x: f"{x:.3f}",
        help="Standard deviation of phase noise"
    )

with col2:
    selected_jitter = st.selectbox(
        "Jitter Noise Level:",
        options=scan_results['jitter_levels'],
        format_func=lambda x: f"{x:.3f}",
        help="Standard deviation of jitter noise"
    )

# Find the corresponding experiment
experiment_key = (selected_phase, selected_jitter)
if experiment_key in scan_results['experiments']:
    experiment_path = scan_results['experiments'][experiment_key]

    # Check experiment validity
    validity = check_experiment_validity(experiment_path)

    # Display experiment info
    st.markdown("### Selected Experiment")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(f"**Path:** `{experiment_path.name}`")

        # Display validity checks
        if validity['is_valid']:
            st.success("✅ All required files present")
        else:
            st.error("❌ Missing required files")

        # Show details
        with st.expander("Experiment Details"):
            st.json({
                "Config exists": validity['config'],
                "Checkpoints exist": validity['checkpoints'],
                "PCA models exist": validity['pca_models'],
                "Number of checkpoints": validity['n_checkpoints'],
                "Number of PCA models": validity['n_pca_models']
            })

    with col2:
        if validity['is_valid']:
            if st.button("🚀 Go to Visualization", type="primary", use_container_width=True):
                st.session_state.selected_experiment = experiment_path.name
                st.session_state.experiment_path = str(experiment_path)
                st.session_state.phase_noise = selected_phase
                st.session_state.jitter_noise = selected_jitter
                st.success("✅ Experiment selected! Navigate to Visualization page.")
                st.balloons()
        else:
            st.error("Cannot proceed - missing files")
            if not validity['pca_models']:
                st.warning("Run `save_pca_models.py` first")
else:
    st.warning(f"No experiment found for phase={selected_phase}, jitter={selected_jitter}")

# Display experiment grid
st.markdown("### Available Experiments")

# Create a grid showing all available experiments
import pandas as pd

# Create matrix of available experiments
matrix_data = []
for phase in scan_results['phase_levels']:
    row = {'Phase': phase}
    for jitter in scan_results['jitter_levels']:
        if (phase, jitter) in scan_results['experiments']:
            row[f'J={jitter:.3f}'] = '✅'
        else:
            row[f'J={jitter:.3f}'] = '❌'
    matrix_data.append(row)

df = pd.DataFrame(matrix_data)
df = df.set_index('Phase')

st.dataframe(df, use_container_width=True)

# Footer info
st.markdown("---")
st.caption("Navigate to the Visualization page after selecting an experiment")
