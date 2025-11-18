# LSTM Beat Prediction with Temporal Noise Analysis

## Overview

This repository contains a comprehensive framework for training and analyzing LSTM networks on beat prediction tasks with varying levels of temporal noise. The system explores how neural networks learn to anticipate rhythmic patterns under different noise conditions, providing deep insights into the internal dynamics through interactive visualizations.

## Key Features

### 🧠 Neural Network Architecture

- LSTM-based models for temporal beat prediction
- Configurable architecture (layers, hidden units, dropout)
- Support for multiple target types (rectangular, gamma, gaussian distributions)

### 🎵 Beat Prediction Task

- Networks learn to predict beats before they occur
- Input: Sequences of isochronous beats with configurable timing
- Output: Anticipatory signals that predict upcoming beats
- Temporal noise injection (phase noise and jitter) to test robustness

### 📊 Analysis Pipeline

- **State Analysis**: Extract and save hidden/cell states during inference
- **PCA Dimensionality Reduction**: Project high-dimensional states to 3D for visualization
- **Interactive Visualization**: Streamlit app for real-time model exploration

## Repository Structure

```
├── models/                 # LSTM model definitions
├── data/                   # Data generators for beat sequences
├── training/               # Training utilities and trainer classes
├── analysis/              
│   ├── save_pca_models.py     # Compute PCA for state space analysis
│   ├── state_analyzer.py      # Extract states from trained models
│   └── interactive_sliders.py # Matplotlib-based visualizations
├── pages/                  # Streamlit app pages
│   ├── 1_Experiment_Selection.py
│   └── 2_Visualization.py
├── utils/                  # Utility functions
├── config/                 # Configuration files
└── app.py                  # Main Streamlit application
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd lstm-beat-prediction

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_ui.txt
```

## Usage

### 1. Training Models

Train models with different noise levels:

```bash
python main.py --train --config config/default_config.json
```

### 2. Analyzing Trained Models

#### Extract States and Compute PCA

For experiments with gaussian noise:

```bash
# Compute PCA models for all checkpoints
python analysis/save_pca_models.py --base_dir experiments/gaussian_noise
```

For standard experiments:

```bash
# Analyze states across checkpoints
python analyze_states.py --experiment_path experiments/[timestamp] --analysis_config config/analysis_config.json

# Preprocess with PCA
python analysis/preprocess_pca.py --experiment_path experiments/[timestamp]
```

### 3. Interactive Visualization

Launch the Streamlit app:

```bash
streamlit run app.py
```

#### Using the Visualization App:

1. **Experiment Selection**
   - Navigate to "Experiment Selection" page
   - Enter the base directory containing experiments
   - Select phase and jitter noise levels from dropdowns
   - Click "Go to Visualization"

2. **Model Interaction**
   - Use the epoch slider to select different training stages
   - Configure input parameters (tempo, pulses, noise levels)
   - Generate pulse sequences or manually specify exact timings
   - Click "Run Model" to execute

3. **Results Analysis**
   - View 2D plots showing input/output signals with inter-pulse intervals
   - Explore 3D PCA-transformed state trajectories
   - Hover over trajectory points to see input/output values
   - Toggle between hidden and cell states
   - Beat markers show where pulses occur in the state space

## Visualization Features

### 📈 2D Signal Plots

- Input and output waveforms
- Beat markers at pulse onset times
- Inter-pulse interval measurements
- Real-time comparison of model predictions

### 🌐 3D State Trajectories

- PCA-reduced neural state visualization
- Interactive rotation, zoom, and pan
- Time-based color gradients
- Beat event markers in state space
- Hover information showing:
  - Time point
  - PCA coordinates (PC1, PC2, PC3)
  - Input value at that timestep
  - Network output at that timestep

### 🎛️ Interactive Controls

- Epoch selection slider
- Noise parameter adjustment
- Manual pulse time editing
- State type selection (hidden/cell)
- Beat marker visibility toggle

## Experiment Organization

Experiments are organized by noise levels:

```
experiments/gaussian_noise/
├── 20251109_160331_p0.005_j0.01/  # Phase=0.005, Jitter=0.01
│   ├── checkpoints/                # Model checkpoints
│   ├── pca_models/                 # PCA transformations
│   └── config.json                 # Experiment configuration
└── ...
```

## Technical Details

### Noise Models

- **Phase Noise**: Cumulative timing drift affecting all subsequent beats
- **Jitter**: Individual random variations per beat
- **Variable pulse counts**: 3-7 pulses per sequence

### PCA Analysis

- Fits shared PCA space for all periods within each epoch
- Captures 3D representations of high-dimensional neural states
- Enables comparison across different temporal conditions

### State Space Interpretation

The 3D trajectories reveal how the network's internal dynamics evolve:

- Anticipatory ramping before beats
- Reset dynamics after beat events
- Period-specific trajectory patterns
- Learning progression across epochs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- NumPy, Matplotlib, Plotly, scikit-learn

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information]