# Merchant Lab Synchronization-Continuation Experiments

This document provides a comprehensive overview of the Merchant Lab's experimental paradigm for studying neural mechanisms of beat synchronization, and explains how our LSTM implementation aligns with their methodology.

## Table of Contents

1. [Background](#background)
2. [Merchant Lab Experimental Setup](#merchant-lab-experimental-setup)
   - [Subjects and Apparatus](#subjects-and-apparatus)
   - [Stimulus Parameters](#stimulus-parameters)
   - [Trial Structure](#trial-structure)
   - [Behavioral Measures](#behavioral-measures)
   - [Neural Recording and Analysis](#neural-recording-and-analysis)
3. [Our LSTM Implementation](#our-lstm-implementation)
   - [Task Design Decisions](#task-design-decisions)
   - [Key Alignments with Merchant Lab](#key-alignments-with-merchant-lab)
   - [Implementation Details](#implementation-details)
4. [Configuration Reference](#configuration-reference)
5. [Future Improvements](#future-improvements)
6. [References](#references)

---

## Background

The ability to perceive and produce rhythmic patterns is fundamental to human behavior, from music and dance to speech and locomotion. Understanding the neural mechanisms underlying beat synchronization requires carefully designed experimental paradigms that separate different aspects of timing: perceiving a beat, synchronizing movements with it, and maintaining the tempo internally.

The Merchant Lab at Universidad Nacional Autónoma de México has developed a sophisticated experimental framework using rhesus macaques to study these mechanisms at the single-neuron level. Their work has revealed fundamental principles about how the brain represents time intervals and generates rhythmic behavior.

---

## Merchant Lab Experimental Setup

### Subjects and Apparatus

**Subjects**: Two rhesus macaque monkeys (Macaca mulatta), one male and one female, aged 8-10 years, weighing 5-10 kg.

**Physical Setup**:
- **Lever/Key**: Where the monkey rests its hand at trial start; detects hand placement and removal
- **Button**: Tapping target during synchronization; records precise tap timing
- **Display**: Screen for visual stimuli presentation
- **Speaker**: For auditory stimulus delivery
- **Reward System**: Juice delivery proportional to trial length

### Stimulus Parameters

| Parameter | Value |
|-----------|-------|
| **Visual Stimulus** | Red square, 5 cm side length |
| **Auditory Stimulus** | White noise burst |
| **Stimulus Duration** | 33 ms (both modalities) |
| **Inter-Stimulus Intervals** | 450 ms and 850 ms (Betancourt et al., 2023) |
| | 450, 550, 650, 750, 850, 950 ms (Dotov et al., 2025) |
| **Timing Noise** | None — stimuli are perfectly periodic |

**Critical Point**: The metronome stimuli have **no added noise or jitter**. Inter-stimulus intervals are fixed and precise within each condition. Variability in the data comes from the monkey's motor execution, not from the stimulus.

### Trial Structure

```
┌─────────────────┬──────────────────────┬─────────────────────────────────┐
│   DELAY/HOLD    │   ATTENTION PHASE    │      SYNCHRONIZATION PHASE      │
│   (1.2-4.0 s)   │   (2-3 stimuli)      │      (5-6 taps / intervals)     │
├─────────────────┼──────────────────────┼─────────────────────────────────┤
│ Hand on lever   │ Stimuli presented    │ Monkey taps button in sync      │
│ Wait for start  │ Monkey holds lever   │ with ongoing metronome          │
│                 │ NO movement allowed  │                                 │
└─────────────────┴──────────────────────┴─────────────────────────────────┘
```

**Phase Details**:

1. **Delay Period** (1.2-4.0 seconds)
   - Monkey places hand on lever
   - Variable duration to prevent anticipation
   - Must maintain position until metronome begins

2. **Attention/Beat Perception Phase** (2-3 stimuli)
   - Metronome begins presenting stimuli
   - Monkey must continue holding the lever
   - NO movement allowed — immediate trial termination if moved
   - Purpose: Allow monkey to perceive and internalize the tempo

3. **Synchronization Phase** (5-6 produced intervals)
   - Monkey releases lever, moves to button, begins tapping
   - Taps must align with ongoing metronome stimuli
   - First produced interval excluded from analysis (transition movement)
   - Metronome continues throughout — this is synchronization, not continuation

### Trial Termination and Reward

**Trial Terminated Without Reward If**:
- Movement during attention phase (early lever release)
- Tapping starts too late after attention phase
- Any tap-stimulus asynchrony exceeds ±200 ms

**Correct Trial Criteria**:
- All produced intervals have error < 18% of target interval
- All asynchronies remain within ±200 ms window

**Reward**: Fruit juice, amount proportional to trial length

### First Produced Interval Exclusion

A critical methodological detail: **the first produced interval is always excluded from analysis**.

**Physical Reason**:

When transitioning from attention to synchronization:
1. Monkey's hand starts on the **lever** (holding position)
2. Monkey must **release** the lever
3. Monkey must **reach** through space (~10-15 cm)
4. Monkey must **land** on the button
5. Only then does normal **tapping** begin

This first "interval" is fundamentally different:

| Aspect | Transition Interval | Normal Tapping Interval |
|--------|---------------------|------------------------|
| Movement type | Reaching (lever → button) | Tapping (button → button) |
| Distance | ~10-15 cm through space | ~1-2 cm vertical |
| Duration | Variable (includes reaction time) | Consistent |
| Kinematics | Ballistic reach | Stereotyped tap |

**Behavioral Data**:
- First interval is systematically longer and more variable
- Including it would contaminate measures of tapping precision
- Standard practice in synchronization-continuation literature

**Neural Data**:
- First interval has different neural dynamics (movement preparation vs. rhythmic tapping)
- Including it would mix different neural processes
- Trajectory analysis assumes homogeneous tapping movements

### Block Design

- **4 conditions**: 2 intervals × 2 modalities
  - Short Auditory (450 ms)
  - Long Auditory (850 ms)
  - Short Visual (450 ms)
  - Long Visual (850 ms)
- **25 trials per block**
- **Order randomized** across recording days
- Within a block, tempo and modality are constant (contextual knowledge)

### Behavioral Measures

| Measure | Definition | Interpretation |
|---------|------------|----------------|
| **Constant Error** | Produced − Instructed duration | Accuracy (bias) |
| **Temporal Variability** | SD of produced intervals | Precision |
| **Lag-1 Autocorrelation** | Correlation between consecutive intervals | Error correction (negative = correcting) |

**Key Finding**: Monkeys show better performance with visual than auditory metronomes (opposite of humans).

### Movement Kinematics

Recorded at 250 fps using high-speed camera:

- **Movement Duration**: Constant ~150-200 ms across all tempos
- **Dwell Time**: Adjusts with tempo (longer intervals = longer dwells)
- **Interpretation**: Tempo is controlled by adjusting wait time between stereotyped movements, not by changing movement speed

### Neural Recording and Analysis

**Recording Setup**:
- 64-site silicon probes per hemisphere (128 total channels)
- Medial premotor cortex (SMA / pre-SMA boundary)
- 24,414 Hz sampling rate
- Up to 142 isolated neurons per session

**Preprocessing**:
1. Spike sorting (custom ABVA algorithm or KiloSort)
2. Gaussian kernel convolution (σ = 10-20 ms) → continuous firing rate
3. **Time normalization**: Each interval mapped to same number of bins regardless of duration

**Dimensionality Reduction**:
- PCA or Gaussian Process Factor Analysis (GPFA)
- Typically 3-6 dimensions retained
- First 3 PCs explain ~15% of variance

**Key Neural Trajectory Properties**:

| Property | Description |
|----------|-------------|
| **Circular Loops** | Trajectories form regenerating cycles, one per produced interval |
| **Tap Separatrix** | Consistent convergence point in state space at tap times |
| **Amplitude Modulation** | Longer intervals → larger trajectory amplitude |
| **Temporal Scaling** | Longer intervals → slower trajectory speed |
| **AMSI Index** | Balance between amplitude and speed encoding (~0 = balanced mixture) |
| **Modality Subspaces** | Auditory vs. visual occupy partially overlapping but distinct regions |

**Neural Sequence Analysis**:
- Poisson-train surprise index identifies cell activation periods
- Cells sorted by peak activation time → sequential "bumps" tiling each interval
- Sequences reset at each tap (regenerating structure)

---

## Our LSTM Implementation

### Task Design Decisions

We designed our LSTM task to match the Merchant Lab paradigm as closely as possible while adapting for computational modeling:

| Aspect | Merchant Lab | Our Implementation | Rationale |
|--------|--------------|-------------------|-----------|
| **Phases** | Attention → Sync | Attention → Sync → **Continuation** | Added continuation to test internal tempo maintenance |
| **Stimulus Timing** | Perfectly periodic | Perfectly periodic | Match exactly — no noise in input |
| **Intervals** | 450, 850 ms (or 450-950 ms) | 450, 550, 650, 750, 850 ms | Discrete steps matching Dotov et al. |
| **Stimulus Duration** | 33 ms | 30 ms | Close match (3 samples at dt=0.01) |
| **Attention Pulses** | 2-3 | 2-3 (randomized) | Match exactly |
| **Sync Pulses** | 5-6 | 5-6 (randomized) | Match exactly |
| **Continuation Pulses** | N/A | 3-5 (randomized) | New — tests tempo without input |
| **First Sync Excluded** | Yes (transition) | Yes (`skip_first_n_sync: 1`) | Match analysis convention |
| **Block Design** | Yes (fixed context) | No (random period per trial) | Increases generalization demand |
| **Feedback Loop** | N/A (motor execution) | Optional feedback system | Simulates sensorimotor feedback |

### Key Alignments with Merchant Lab

#### 1. No Stimulus Timing Noise

```json
"noise_params": {
  "phase_noise_std": 0.0,
  "jitter_std": 0.0
}
```

The metronome is perfectly periodic. This is **critical** — Merchant Lab stimuli have no jitter.

#### 2. Discrete Period Values

```json
"min_period": 0.450,
"max_period": 0.850,
"period_step": 0.100
```

Training uses 5 discrete periods: 450, 550, 650, 750, 850 ms. This matches Dotov et al. (2025) who used intervals from 450-950 ms.

#### 3. Phase Structure

```json
"attention_phase": { "min_n_pulses": 2, "max_n_pulses": 3 },
"sync_phase": { "min_n_pulses": 5, "max_n_pulses": 6 },
"continuation_phase": { "min_n_pulses": 3, "max_n_pulses": 5 }
```

- **Attention**: 2-3 pulses (matches "first two or three stimuli")
- **Sync**: 5-6 pulses (matches "five to six rhythmic intervals")
- **Continuation**: 3-5 pulses (our addition for testing internal timing)

#### 4. First Sync Interval Excluded

```json
"skip_first_n_sync": 1,
"ignore_skipped_sync_error": true
```

**Why Merchant Lab Excludes the First Interval**:

In the physical experiment, when the synchronization phase begins:
1. The monkey's hand is resting on the **lever**
2. The monkey must release the lever, reach through space, and land on the **button**
3. Only then can normal tapping begin

This first "interval" includes a **transition movement** (lever → button) that is fundamentally different from subsequent tapping movements:
- It's a reaching movement, not a tapping movement
- It includes reaction time to initiate movement
- It's longer and more variable than actual tapping intervals

Therefore, Merchant Lab **excludes this first produced interval from all behavioral and neural analysis**.

**How We Implement This**:

We use two independent parameters:

| Parameter | What It Does |
|-----------|--------------|
| `skip_first_n_sync: 1` | Don't generate a target for the first sync pulse (target stays at baseline) |
| `ignore_skipped_sync_error: true` | Don't compute loss during the first sync pulse period |

**Why Two Parameters?**

- `skip_first_n_sync` controls **target generation** — no prediction target exists for the first sync pulse
- `ignore_skipped_sync_error` controls **loss computation** — the network isn't penalized regardless of what it outputs

Without `ignore_skipped_sync_error`, the network would still be penalized for outputting anything ≠ 0 during the first sync pulse (since target = baseline = 0). With it enabled, the network is free to output anything during this transition period, just like the monkey is free to move however it needs to reach the button.

**Visual Representation**:

```
Phase:        [  ATTENTION  ]  [ SKIP ][     SYNC (analyzed)    ]  [ CONTINUATION ]
Input:            ●   ●   ●      ●        ●    ●    ●    ●            -   -   -
Target:           0   0   0      0        ↑    ↑    ↑    ↑            ↑   ↑   ↑
Loss Mask:        [masked?]    [masked]   [  computed  ]              [computed]
                                  ^
                           Transition interval
                           (excluded from analysis)
```

This precisely mirrors how Merchant Lab handles their data: the transition interval exists in the data but is excluded from analysis.

#### 5. Phase-Specific Loss Computation

```json
"ignore_attention_error": true,
"ignore_skipped_sync_error": true,
"ignore_tail_error": true
```

We provide **independent control** over loss masking for each phase:

| Parameter | Phase Masked | Merchant Lab Equivalent |
|-----------|--------------|------------------------|
| `ignore_attention_error` | Attention phase | Monkey holds lever, no tapping analyzed |
| `ignore_skipped_sync_error` | First N sync pulses | Transition interval excluded |
| `ignore_tail_error` | After continuation ends | Trial ended, no data collected |

**Why Independent Control?**

Different research questions may require different masking strategies:

| Scenario | `ignore_attention` | `ignore_skipped_sync` | `ignore_tail` |
|----------|--------------------|-----------------------|---------------|
| Full Merchant alignment | ✓ | ✓ | ✓ |
| Study attention-phase dynamics | ✗ | ✓ | ✓ |
| Study transition behavior | ✓ | ✗ | ✓ |
| No masking (baseline) | ✗ | ✗ | ✗ |

**Implementation Detail**:

Loss masking works by setting `output = target` for masked regions before computing MSE loss. This means:
- Masked regions contribute zero to the loss
- Gradients don't flow from masked regions
- Network is free to output anything in masked regions

#### 6. Variability Through Randomization

Instead of timing noise, variability comes from:
- Random number of pulses per phase (Poisson-like sampling)
- Random period selection from discrete set
- Random start offset within first period

This prevents the network from memorizing specific sequences while maintaining precise stimulus timing.

### Implementation Details

#### Sequence Generation

```
Time →
        |----Attention----|--Skip--|------Sync (analyzed)------|----Continuation----|--Tail--|

Input:    ●    ●    ●        ●         ●    ●    ●    ●             0    0    0    0      0
Target:   0    0    0        0         ↑    ↑    ↑    ↑             ↑    ↑    ↑    ↑      0
Loss:   [optional mask]   [masked]   [  computed  ]              [  computed  ]        [masked]
                              ^
                        Transition
                        (like lever→button)
```

- **Input**: Rectangular pulses (30 ms width) during attention, skip, and sync phases only
- **Target**: Gaussian anticipation peaks during sync (after skip) and continuation
- **Loss Mask**: Configurable per phase; sync (after skip) and continuation always contribute

**Mapping to Merchant Lab**:

| Our Implementation | Merchant Lab Equivalent |
|--------------------|------------------------|
| Attention phase (input, no target) | Monkey attends to 2-3 stimuli while holding lever |
| Skipped sync pulse (input, no target, masked) | First produced interval (transition movement) |
| Sync phase (input + target) | Monkey taps in synchrony with metronome |
| Continuation phase (no input, target) | (Our addition) Tests internal tempo maintenance |
| Tail (no input, no target, masked) | Inter-trial interval |

#### Period Sampling

```python
def sample_period(task_config):
    if "period_step" in task_config:
        min_p = task_config["min_period"]
        max_p = task_config["max_period"]
        step = task_config["period_step"]
        values = np.arange(min_p, max_p + step/2, step)
        return np.random.choice(values)
```

With `period_step: 0.1`, this produces: [0.45, 0.55, 0.65, 0.75, 0.85]

#### Pulse Count Sampling

```python
def sample_n_pulses(min_n, max_n):
    if min_n == max_n:
        return min_n
    lambda_param = (min_n + max_n) / 2
    while True:
        sample = np.random.poisson(lambda_param)
        if min_n <= sample <= max_n:
            return sample
```

Uses truncated Poisson distribution centered at midpoint.

#### Feedback Loop (Optional)

When feedback is enabled, the model receives a delayed copy of its own output:

```
Output → [Threshold Detection] → [Delay Buffer] → [Add to Input]
                                      ↓
                               [Pulse Generator]
```

This simulates the auditory/proprioceptive feedback a subject receives from their own motor actions. The `continuation_decay` parameter models diminishing feedback confidence without external reference signals.

**Feedback Parameters:**
- `threshold`: Output level that triggers feedback (e.g., 0.1)
- `delay`: Time delay before feedback injection (e.g., 50ms)
- `pulse_shape`: Shape of feedback pulse (rectangular, gaussian, gamma)
- `continuation_decay`: Exponential decay rate during continuation phase

---

## Configuration Reference

### Full Configuration File

```json
{
  "paths": {
    "experiment_dir": "./experiments/sync_continuation",
    "experiment_path": null
  },
  "model": {
    "num_layers": 1,
    "hidden_size": 32,
    "dropout": 0,
    "input_size": 1,
    "output_size": 1
  },
  "training": {
    "epochs": 10000,
    "batch_size": 50,
    "learning_rate": 0.005,
    "trials_per_epoch": 1000,
    "validation_trials": 100,
    "checkpoint_best_only": true,
    "optimizer": "adam",
    "use_scheduler": false
  },
  "task": {
    "task_type": "sync_continuation",
    
    "min_period": 0.450,
    "max_period": 0.850,
    "period_step": 0.100,
    
    "attention_phase": {
      "min_n_pulses": 2,
      "max_n_pulses": 3
    },
    "sync_phase": {
      "min_n_pulses": 5,
      "max_n_pulses": 6
    },
    "continuation_phase": {
      "min_n_pulses": 3,
      "max_n_pulses": 5
    },
    
    "target_shape": "gaussian",
    
    "pulse_width": 0.03,
    "pulse_height": 1.0,
    "dt": 0.01,
    "sequence_length": 10.0,
    "baseline_value": 0.0,
    "output_offset": 0.0,
    
    "noise_params": {
      "phase_noise_std": 0.0,
      "jitter_std": 0.0
    },
    
    "gaussian_params": {
      "gaussian_length": 0.05,
      "gaussian_sigma": null,
      "gaussian_max_height": 1.0
    },
    
    "skip_first_n_sync": 1,
    "ignore_skipped_sync_error": true,
    "ignore_attention_error": true,
    "ignore_tail_error": true
  },
  "logging": {
    "tensorboard": true,
    "log_interval": 10,
    "save_interval": 100
  },
  "testing": {
    "n_test_periods": 9,
    "test_trials_per_period": 10,
    "test_period_step": 0.050
  },
  "feedback": {
    "enabled": true,
    "threshold": 0.1,
    "delay": 0.05,
    "pulse_shape": "rectangular",
    "pulse_width": 0.05,
    "pulse_height": 1.0,
    "refractory_period": 0.05,
    "continuation_decay": 0.9
  },
  "seed": 42
}
```

### Parameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `task_type` | `"sync_continuation"` | Enables three-phase task |
| `min_period` | 0.450 | Shortest interval (seconds) |
| `max_period` | 0.850 | Longest interval (seconds) |
| `period_step` | 0.100 | Discrete step size for period sampling |
| `pulse_width` | 0.03 | Stimulus duration (30 ms ≈ Merchant's 33 ms) |
| `dt` | 0.01 | Time resolution (10 ms) |
| `sequence_length` | 10.0 | Total sequence duration (seconds) |
| `skip_first_n_sync` | 1 | Skip first sync target (transition interval) |
| `ignore_skipped_sync_error` | true | Don't compute loss during skipped sync |
| `ignore_attention_error` | true | Don't compute loss during attention |
| `ignore_tail_error` | true | Don't compute loss after continuation |
| `test_period_step` | 0.050 | Finer step for testing generalization |
| `feedback.enabled` | true/false | Enable/disable feedback loop |
| `feedback.threshold` | 0.1 | Output level triggering feedback |
| `feedback.delay` | 0.05 | Feedback delay in seconds |
| `feedback.continuation_decay` | 0.9 | Pulse height decay coefficient in continuation phase |

---

## Future Improvements

### Implemented Features

The following features have been implemented:

- **Feedback Loop**: Configurable sensorimotor feedback system with delay, threshold, pulse shapes, and continuation phase decay
- **Animated Visualization**: Real-time 3D trajectory playback with Play/Pause, speed control, and timeline scrubbing
- **Ideal Beat Markers**: Visual reference lines showing where perfect beats would occur, extending through the entire experiment

### 1. Neural Trajectory Analysis Measures

Implement Merchant Lab analysis metrics:

[//]: # ()
[//]: # (```python)

[//]: # (# Planned: analysis/merchant_measures.py)

[//]: # ()
[//]: # (def compute_trajectory_amplitude&#40;hidden_states, anchor_point&#41;:)

[//]: # (    """Euclidean distance from anchor to each state.""")

[//]: # (    pass)

[//]: # ()
[//]: # (def compute_tap_separatrix&#40;hidden_states, tap_indices&#41;:)

[//]: # (    """Find convergence region at tap times.""")

[//]: # (    pass)

[//]: # ()
[//]: # (def compute_amsi_index&#40;hidden_states, short_period, long_period&#41;:)

[//]: # (    """Balance between amplitude and speed encoding.""")

[//]: # (    pass)

[//]: # ()
[//]: # (def compute_oscillation_index&#40;hidden_states, period&#41;:)

[//]: # (    """Strength of cyclical structure.""")

[//]: # (    pass)

[//]: # (```)

### 2. Generalizability Analysis

Following Merchant Lab methodology:

- Euclidean distance matrices between neural sequences
- Diagonal asymmetry index for cross-condition generalization
- Identify neurons with condition-specific vs. shared activation

### 3. Serial Order Effects

Analyze how neural representations change across sequential intervals:

- Which hidden units show serial order sensitivity?
- Does this differ between "generalized" and "over-specialized" solutions?

### 4. Modality Conditions

Add support for multiple "modalities" (distinct input channels):

```json
"modality": "visual",  // or "auditory"
```

This would allow testing whether the network develops distinct subspaces for different input types, as seen in Merchant Lab data.

### 5. Time Normalization for Analysis

Implement Merchant Lab's time normalization:

- Map each produced interval to fixed number of bins
- Enables direct comparison of trajectories across tempos
- Reveals relative vs. absolute time encoding

### 6. Movement Kinematics Proxy

The network's output could be analyzed for:

- "Movement duration" (width of output peaks)
- "Dwell time" (gap between peaks)
- Test if network shows similar tempo control strategy to monkeys

### 7. Block Design Option

Add option for blocked training (fixed tempo within blocks):

```json
"block_design": {
  "enabled": true,
  "trials_per_block": 25
}
```

This would match Merchant Lab's contextual knowledge manipulation.

---

## References

### Primary Merchant Lab Papers

1. **Betancourt et al. (2023)** - [LINK]
   - Main synchronization task methodology
   - Neural trajectory analysis (tap separatrix, AMSI index)
   - Neural sequence analysis

2. **Dotov et al. (2025)** - [LINK]
   - Attention-then-synchronization task
   - Resonance-like amplitude buildup during attention
   - Oscillation index predicts performance
