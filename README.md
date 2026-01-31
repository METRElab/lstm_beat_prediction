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

### 3. Visualize Results

```bash
streamlit run app.py
```

## Repository Structure

```
├── models/
│   └── lstm_model.py           # LSTM architecture
├── data/
│   └── generators.py           # Sequence generation (sync-continuation task)
├── training/
│   └── trainer.py              # Training loop with phase-aware loss masking
├── analysis/
│   ├── save_pca_models.py      # PCA for state space analysis
│   └── state_analyzer.py       # Hidden state extraction
├── utils/
│   ├── visualization.py        # Plotting functions
│   └── viz_utils.py            # 3D trajectory visualization
├── pages/                      # Streamlit app pages
├── app.py                      # Main Streamlit application
├── main.py                     # Training/testing entry point
├── training_config_sync_continuation.json  # Default config
├── README.md                   # This file
└── MERCHANT_EXPERIMENT.md      # Detailed experiment documentation
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

## Training vs Testing Periods

- **Training**: 5 discrete periods (450, 550, 650, 750, 850 ms)
- **Testing**: 9 periods including interpolation (450, 500, 550, ..., 850 ms)

This tests generalization to untrained tempos.

## Visualization Features

### 2D Signal Plots
- Input/output waveforms with phase boundaries
- Phase shading (attention=blue, sync=green, continuation=orange)
- Beat markers and timing annotations

### 3D State Trajectories
- PCA-reduced hidden state visualization
- Time-based color gradients
- Interactive rotation and zoom
- Beat event markers

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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Plotly
- Streamlit (for visualization)
- scikit-learn (for PCA)

## License

[Your license information]