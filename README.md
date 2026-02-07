# LSTM Beat Prediction: Synchronization-Continuation Task

A computational framework for training and analyzing LSTM networks on rhythmic timing tasks, inspired by the Merchant Lab's work on neural mechanisms of beat synchronization in primates.

## Overview

This repository implements an LSTM-based model that learns to:
1. **Attend** to rhythmic stimuli without responding
2. **Synchronize** predictions with ongoing beats
3. **Continue** predicting beats after stimuli stop

This three-phase structure mirrors the synchronization-continuation paradigm used in primate neurophysiology research to study internal timing mechanisms.

> 📖 **For detailed information about the Merchant Lab experiments and how this implementation aligns with their methodology, see [MERCHANT_EXPERIMENT.md](MERCHANT_EXPERIMENT.md)**

## Key Features

### Task Structure

```
[Attention Phase]     [Synchronization Phase]    [Continuation Phase]     [Tail]
     ●  ●  ●              ●  ●  ●  ●  ●               (no input)           ...
   Input only          Input + Target              Target only           Ignored
   (learn tempo)       (predict beats)          (maintain tempo)
```

### Neural Network

- LSTM-based architecture for temporal sequence learning
- Configurable hidden size, layers, and dropout
- Step-by-step state extraction for trajectory analysis

### Analysis Pipeline

- PCA dimensionality reduction of hidden states
- Interactive 3D trajectory visualization
- Phase-aware performance metrics

### Feedback Loop System

- Optional feedback loop that injects pulses back into the model when output crosses a threshold
- Configurable delay, pulse shape, and refractory period
- Continuation phase decay: feedback strength can decay exponentially during self-paced continuation

## Installation

```bash
git clone <repository-url>
cd lstm-beat-prediction

pip install -r requirements.txt
pip install -r requirements_ui.txt  # For visualization app
```

## Quick Start

### 1. Train a Model

```bash
python main.py --train --config training_config_sync_continuation.json
```

### 2. Test a Trained Model

```bash
python main.py --test --config experiments/sync_continuation/<timestamp>/config.json --checkpoint checkpoint_epoch_XXXX.pth
```

### 3. Analyze and Visualize Results

After training completes, run the analysis pipeline:

```bash
# Step 1: Extract hidden/cell states for all checkpoints
python analyze_states.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --analysis_config config/pca_analysis_config.json

# Step 2: Compute PCA models for dimensionality reduction
python analysis/save_pca_models.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --analysis_config config/pca_analysis_config.json

# Step 3: Preprocess PCA states for interactive visualization
python analysis/preprocess_pca.py \
    --experiment_path experiments/sync_continuation/<timestamp>
```

Then choose a visualization method:

#### Interactive Sliders (Matplotlib)

Explore trajectories across epochs and periods with sliders. Shows input, target, output signals and 3D state trajectories with phase shading.

```bash
# Hidden states
python analysis/interactive_sliders.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot_type hidden

# Cell states
python analysis/interactive_sliders.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot_type cell

# MSE progression across epochs
python analysis/interactive_sliders.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot_type error

# All visualizations
python analysis/interactive_sliders.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot_type all
```

**Animation Controls** (in interactive_sliders.py):
- **Play/Pause**: Click to start/stop trajectory animation
- **Speed slider**: Adjust playback speed (default 50ms/frame)
- **Timeline slider**: Scrub to any frame in the sequence
- **Ideal beat lines**: Gray dashed lines show where perfect beats would occur

#### Sync-Continuation Specific Plots (Matplotlib)

Standalone plots designed for the sync-continuation task with phase boundary visualization.

```bash
# All plots for latest epoch
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot all --save

# Input/output with phase shading for a specific epoch and period
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot output --epoch 5000 --period_idx 0

# 3D trajectories colored by period (all periods overlaid)
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot trajectories --epoch 5000 --state_type hidden

# Single trajectory colored by phase (attention/sync/continuation)
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot phases --epoch 5000 --period_idx 0 --state_type hidden

# MSE vs period across multiple epochs
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot mse

# Training progression (how trajectory evolves over epochs)
python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot progression --period_idx 0 --state_type hidden
```

#### TensorBoard (Training Curves)

```bash
tensorboard --logdir experiments/sync_continuation/
```

Then open `http://localhost:6006` in your browser.

## Repository Structure

```
├── models/
│   ├── lstm_model.py                  # LSTM architecture
│   ├── feedback.py                    # Feedback buffer and pulse generator
│   └── feedback_lstm.py               # Feedback-enabled LSTM wrapper
├── data/
│   └── generators.py                  # Sequence generation (sync-continuation + legacy)
├── training/
│   ├── trainer.py                     # Training loop with phase-aware loss masking
│   └── feedback_trainer.py            # Trainer for feedback-enabled models
├── analysis/
│   ├── save_pca_models.py             # PCA model computation for all checkpoints
│   ├── state_analyzer.py              # Hidden/cell state extraction with phase info
│   ├── preprocess_pca.py              # PCA preprocessing for interactive visualization
│   ├── interactive_sliders.py         # Interactive matplotlib plots with epoch/period sliders
│   └── batch_process_pca.py           # Batch PCA processing utility
├── utils/
│   ├── visualization.py               # Static plotting functions
│   ├── viz_utils.py                   # 3D trajectory visualization (Plotly)
│   ├── model_runner.py                # Model loading and inference utility
│   ├── input_generator.py            # Manual input generation utility
│   └── logging.py                     # Logging setup
├── config/
│   └── pca_analysis_config.json       # Analysis configuration
├── pages/                             # Streamlit app pages (legacy)
├── app.py                             # Streamlit application (legacy)
├── main.py                            # Training/testing entry point
├── analyze_states.py                  # State analysis entry point
├── visualize_sync_continuation.py     # Sync-continuation visualization script
├── training_config_sync_continuation.json  # Default sync-continuation config
├── README.md                          # This file
└── MERCHANT_EXPERIMENT.md             # Detailed experiment documentation
```

## Configuration

Key parameters in `training_config_sync_continuation.json`:

```json
{
  "task": {
    "task_type": "sync_continuation",
    
    "min_period": 0.450,
    "max_period": 0.850,
    "period_step": 0.100,
    
    "attention_phase": { "min_n_pulses": 2, "max_n_pulses": 3 },
    "sync_phase": { "min_n_pulses": 5, "max_n_pulses": 6 },
    "continuation_phase": { "min_n_pulses": 3, "max_n_pulses": 5 },
    
    "skip_first_n_sync": 1,
    "ignore_skipped_sync_error": true,
    "ignore_attention_error": true,
    "ignore_tail_error": true
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `period_step` | Discrete tempo intervals (e.g., 450, 550, 650, 750, 850 ms) |
| `attention_phase` | Pulses to observe before predicting |
| `sync_phase` | Pulses to predict with input present |
| `continuation_phase` | Pulses to predict without input |
| `skip_first_n_sync` | First sync pulses excluded from targets (transition) |
| `ignore_skipped_sync_error` | Don't penalize outputs during skipped sync pulses |
| `ignore_attention_error` | Don't penalize outputs during attention |
| `ignore_tail_error` | Don't penalize outputs after continuation |

## Feedback System

The model supports an optional feedback loop that simulates sensorimotor feedback. When the model's output crosses a threshold, a feedback pulse is injected back into the input after a configurable delay.

### Feedback Configuration

Add a `feedback` section to your config file:

```json
{
  "feedback": {
    "enabled": true,
    "threshold": 0.1,
    "delay": 0.05,
    "pulse_shape": "rectangular",
    "pulse_width": 0.05,
    "pulse_height": 1.0,
    "refractory_period": 0.05,
    "continuation_decay": 0.9
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `enabled` | Toggle feedback on/off |
| `threshold` | Output value that triggers feedback pulse |
| `delay` | Delay (seconds) between threshold crossing and feedback injection |
| `pulse_shape` | Shape of feedback pulse: "rectangular", "gaussian", or "gamma" |
| `pulse_width` | Duration of feedback pulse (seconds) |
| `pulse_height` | Initial amplitude of feedback pulse |
| `refractory_period` | Minimum time (seconds) between consecutive triggers |
| `continuation_decay` | Decay coefficient (0.0-1.0) for pulse height in continuation phase |

### Continuation Phase Decay

When `continuation_decay < 1.0`, feedback pulse height decays exponentially during the continuation phase:

- **Attention/Sync phases**: Full pulse height (no decay)
- **Continuation phase**: Each trigger multiplies height by `continuation_decay`

Example with `continuation_decay: 0.9`:
- 1st trigger in continuation: height = 1.0
- 2nd trigger: height = 0.9
- 3rd trigger: height = 0.81
- etc.

This simulates diminishing sensory feedback when external stimuli are absent.

## Training vs Testing Periods

- **Training**: 5 discrete periods (450, 550, 650, 750, 850 ms)
- **Testing**: 9 periods including interpolation (450, 500, 550, ..., 850 ms)

This tests generalization to untrained tempos.

## Visualization Features

### Interactive Sliders
- Epoch and period selection via sliders
- Input, feedback, and output signal display with phase shading
- Target vs output comparison
- 3D PCA-reduced hidden/cell state trajectories
- Phase boundary markers (attention → sync → continuation)
- **Ideal beat markers**: Faded dashed lines showing where beats should occur, extending through the entire experiment
- **Animated playback**: Play/Pause button to watch trajectory evolve in real-time
- **Speed control**: Adjust animation speed (10-200ms per frame)
- **Timeline scrubbing**: Jump to any point in the trajectory

### Sync-Continuation Plots
- **Output with Phases**: Input, target, and output signals with color-coded phase regions
- **Trajectories by Period**: All period trajectories overlaid in 3D for comparison
- **Trajectory by Phase**: Single trajectory colored by phase (attention=blue, skipped=red, sync=green, continuation=orange)
- **MSE by Period**: Performance across different tempos over training
- **Training Progression**: How trajectories evolve across epochs

### Phase Color Coding

| Color | Phase | Description |
|-------|-------|-------------|
| 🔵 Blue | Attention | Input present, no prediction expected |
| 🔴 Red | Skipped Sync | First sync pulse, excluded from analysis |
| 🟢 Green | Synchronization | Input + prediction (core task) |
| 🟠 Orange | Continuation | No input, prediction only (tempo test) |
| ⬜ Gray | Tail | After continuation, errors ignored |

## Technical Details

### Phase-Aware Loss Masking

The trainer supports selective loss computation with independent control over each phase:

| Option | What It Masks | Rationale |
|--------|---------------|-----------|
| `ignore_attention_error` | Attention phase | Network learns tempo but we don't care about exact output |
| `ignore_skipped_sync_error` | First N sync pulses | Matches Merchant Lab exclusion of transition interval |
| `ignore_tail_error` | After continuation ends | No meaningful prediction expected |

- **Attention phase**: Optionally ignored (network learns to wait)
- **Skipped sync pulses**: Optionally ignored (transition interval, following Merchant Lab)
- **Sync phase** (after skipped): Always computed (core prediction task)
- **Continuation phase**: Always computed (tests tempo maintenance)
- **Tail**: Optionally ignored (after all targets end)

### Discrete Period Sampling

Periods are sampled from discrete values (not continuous) to match Merchant Lab methodology:

```python
# With period_step = 0.1
# Training periods: [0.45, 0.55, 0.65, 0.75, 0.85]
```

## References

- See [MERCHANT_EXPERIMENT.md](MERCHANT_EXPERIMENT.md) for full citations and methodology details

## Full Workflow

```bash
# 1. Train model
python main.py --train --config training_config_sync_continuation.json

# 2. Test model
python main.py --test \
    --config experiments/sync_continuation/<timestamp>/config.json \
    --checkpoint checkpoint_epoch_XXXX.pth

# 3. Extract states
python analyze_states.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --analysis_config config/pca_analysis_config.json

# 4. Compute PCA
python analysis/save_pca_models.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --analysis_config config/pca_analysis_config.json

# 5. Preprocess for interactive visualization
python analysis/preprocess_pca.py \
    --experiment_path experiments/sync_continuation/<timestamp>

# 6. Visualize (choose one)
python analysis/interactive_sliders.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot_type all

python visualize_sync_continuation.py \
    --experiment_path experiments/sync_continuation/<timestamp> \
    --plot all --save

tensorboard --logdir experiments/sync_continuation/
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Plotly
- scikit-learn (for PCA)
- Streamlit (optional, for legacy visualization)

## References

- See [MERCHANT_EXPERIMENT.md](MERCHANT_EXPERIMENT.md) for full citations and methodology details

## License

[Your license information]

## Citation

```
[Your citation information]
```