"""
Streamlit application for interactive LSTM beat prediction analysis with noise.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="LSTM Beat Prediction Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page - Experiment Selection
st.title("🧠 LSTM Beat Prediction Analysis")
st.markdown("### Interactive Visualization of Models Trained with Different Noise Levels")

# Initialize session state
if 'selected_experiment' not in st.session_state:
    st.session_state.selected_experiment = None
if 'experiment_path' not in st.session_state:
    st.session_state.experiment_path = None

st.markdown("""
This application allows you to:
1. **Select an experiment** based on phase and jitter noise levels
2. **Generate custom inputs** with controllable noise
3. **Visualize model outputs** and internal states in 3D PCA space
4. **Compare performance** across different epochs
""")

st.info("👈 Start by navigating to **Experiment Selection** in the sidebar")

# Add information about the current selection
if st.session_state.selected_experiment:
    st.success(f"✅ Currently selected: {st.session_state.selected_experiment}")
    st.markdown(f"Path: `{st.session_state.experiment_path}`")
else:
    st.warning("⚠️ No experiment selected. Please go to **Experiment Selection** page.")

# Display some instructions
with st.expander("📖 How to use this application"):
    st.markdown("""
    ### Step 1: Select Experiment
    - Navigate to **Experiment Selection** page using the sidebar
    - Choose the base directory containing your experiments
    - Select phase and jitter noise levels from the dropdowns
    - Click "Go to Visualization" to proceed

    ### Step 2: Configure and Run Model
    - Navigate to **Visualization** page
    - Select an epoch using the slider
    - Configure input parameters (tempo, pulses, noise)
    - Generate or manually edit pulse times
    - Click "Run Model" to see results

    ### Step 3: Analyze Results
    - View the input and output signals
    - Explore 3D PCA-transformed states
    - Toggle between hidden and cell states
    - Rotate and zoom the 3D visualization
    """)

# Footer
st.markdown("---")
st.markdown("*Built for analyzing LSTM dynamics under different noise conditions*")
