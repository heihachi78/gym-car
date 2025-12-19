# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project that trains an LSTM-PPO (Proximal Policy Optimization with Long Short-Term Memory) agent to drive in the CarRacing-v3 Gymnasium environment. The agent learns from edge-detected visual observations processed through a CNN-LSTM architecture.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train a new model (16 parallel environments, ~2M steps)
python train.py

# Train with custom settings
python train.py --total-timesteps 1000000 --num-envs 8 --learning-rate 3e-4

# Resume training from checkpoint
python train.py --checkpoint checkpoints/model_latest.pt

# Play with trained model (visual mode)
python main.py
python main.py --checkpoint checkpoints/model_500000.pt
python main.py --random  # Random actions for comparison

# Evaluate model performance
python evaluate.py --checkpoint checkpoints/model_latest.pt --episodes 10

# Monitor training with TensorBoard
tensorboard --logdir runs/
```

## Architecture

### Directory Structure

```
gym-car/
├── train.py                    # Main training script with vectorized envs
├── main.py                     # Inference/play with trained model
├── evaluate.py                 # Model evaluation over multiple episodes
├── models/                     # Neural network components
│   ├── actor_critic.py         # ActorCriticLSTM (main network)
│   ├── cnn_feature_extractor.py # CNN for spatial features
│   └── lstm_state.py           # LSTMState dataclass for hidden states
├── training/                   # Training infrastructure
│   ├── ppo_trainer.py          # PPO algorithm implementation
│   └── rollout_buffer.py       # Experience buffer with GAE computation
├── agents/                     # Agent wrappers
│   └── lstm_ppo_agent.py       # Inference wrapper for trained models
├── wrappers/                   # Custom Gymnasium observation wrappers
│   ├── crop_observation.py     # Remove dashboard from frame
│   ├── sharpen_observation.py  # Image sharpening filter
│   ├── edge_observation.py     # Canny edge detection
│   └── render_observation.py   # Debug visualization
├── utils/
│   └── env_factory.py          # Environment creation and wrapper pipeline
├── checkpoints/                # Saved model checkpoints
└── runs/                       # TensorBoard logs
```

### Data Flow

1. **Observation Pipeline** (in `utils/env_factory.py`):
   ```
   Raw RGB (96×96×3) → Crop (80×96) → Grayscale → Sharpen → Edge Detection (Canny) → Resize (40×48) → Normalize
   ```

2. **Network Architecture** (`models/actor_critic.py`):
   ```
   Observation (40×48) → CNN Feature Extractor (256 features) → LSTM (512 hidden) → Actor head (5 actions) + Critic head (1 value)
   ```

3. **Training Loop** (`train.py`):
   - Collects 2048 steps across 16 parallel environments (32K transitions per update)
   - Computes GAE advantages in `RolloutBuffer`
   - Performs 4 PPO epochs with sequence-aware batching for LSTM

### Key Implementation Details

- **LSTM State Handling**: Hidden states are stored per-step in the buffer and reset on episode boundaries. The `LSTMState` dataclass bundles (h, c) tensors.

- **Normalization Statistics**: The `NormalizeObservation` wrapper maintains running mean/variance that must be saved with checkpoints and restored during inference. This is critical—without it, the model performs poorly.

- **Vectorized Training**: Uses `SyncVectorEnv` with 16 parallel environments. State (obs, hidden, episode rewards) persists across rollout collections.

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-envs` | 16 | Parallel environments |
| `num-steps` | 2048 | Steps per rollout per env |
| `seq-len` | 32 | LSTM sequence length for batching |
| `hidden-size` | 512 | LSTM hidden dimension |
| `learning-rate` | 2.5e-4 | Adam learning rate (with linear decay) |
| `entropy-coef` | 0.03 | Entropy bonus for exploration |

## Documentation

- [learn.md](learn.md) - Comprehensive guide to the RL concepts, PPO algorithm, GAE, and code structure (in Hungarian)
- [review.md](review.md) - Code review with improvement recommendations
