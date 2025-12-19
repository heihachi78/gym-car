# CarRacing LSTM-PPO Agent

A reinforcement learning agent that learns to drive in the [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) environment using Proximal Policy Optimization (PPO) with an LSTM-based neural network.

![CarRacing-v3](https://gymnasium.farama.org/_images/car_racing.gif)

## Features

- **LSTM-PPO Architecture**: Combines CNN feature extraction with LSTM for temporal reasoning
- **Edge Detection Preprocessing**: Uses Canny edge detection to simplify visual input
- **Vectorized Training**: Parallel environment execution for faster training
- **TensorBoard Logging**: Real-time training metrics visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gym-car.git
cd gym-car

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Quick Start

### Train a Model

```bash
# Start training (default: 15M timesteps, 16 parallel environments)
python train.py

# Train with custom settings
python train.py --total-timesteps 2000000 --num-envs 8

# Resume from checkpoint
python train.py --checkpoint checkpoints/model_latest.pt
```

### Play with Trained Model

```bash
# Run the trained agent
python main.py

# Use a specific checkpoint
python main.py --checkpoint checkpoints/model_1000000.pt

# Compare with random actions
python main.py --random
```

### Evaluate Performance

```bash
# Evaluate over multiple episodes
python evaluate.py --checkpoint checkpoints/model_latest.pt --episodes 10
```

### Monitor Training

```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in your browser
```

## Architecture

### Observation Pipeline

The raw 96×96 RGB image is processed through several transformations:

```
Raw RGB (96×96×3) → Crop → Grayscale → Sharpen → Edge Detection → Resize (40×48) → Normalize
```

### Neural Network

```
┌─────────────────┐
│  Observation    │  (40×48 edge-detected image)
└────────┬────────┘
         ▼
┌─────────────────┐
│  CNN Encoder    │  3 conv layers → 256 features
└────────┬────────┘
         ▼
┌─────────────────┐
│     LSTM        │  512 hidden units, temporal memory
└────────┬────────┘
         ▼
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Actor │ │Critic │
└───┬───┘ └───┬───┘
    ▼         ▼
 5 actions   Value
```

### Action Space

| Action | Description |
|--------|-------------|
| 0 | Do nothing |
| 1 | Steer left |
| 2 | Steer right |
| 3 | Accelerate |
| 4 | Brake |

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total-timesteps` | 15,000,000 | Total training steps |
| `--num-envs` | 16 | Parallel environments |
| `--num-steps` | 2048 | Steps per rollout |
| `--learning-rate` | 2.5e-4 | Initial learning rate |
| `--hidden-size` | 512 | LSTM hidden size |
| `--seq-len` | 32 | Sequence length for LSTM batching |

Run `python train.py --help` for all options.

## Project Structure

```
gym-car/
├── train.py              # Training script
├── main.py               # Play with trained model
├── evaluate.py           # Evaluation script
├── models/               # Neural network modules
│   ├── actor_critic.py   # Main LSTM-PPO network
│   ├── cnn_feature_extractor.py
│   └── lstm_state.py
├── training/             # Training infrastructure
│   ├── ppo_trainer.py    # PPO algorithm
│   └── rollout_buffer.py # Experience buffer + GAE
├── wrappers/             # Observation preprocessing
│   ├── crop_observation.py
│   ├── edge_observation.py
│   └── sharpen_observation.py
├── utils/
│   └── env_factory.py    # Environment creation
├── checkpoints/          # Saved models
└── runs/                 # TensorBoard logs
```

## Results

Training typically achieves:
- **~500 reward** after 1M steps
- **~700-800 reward** after 5M steps
- **~850+ reward** after 10M+ steps

A score above 900 indicates near-optimal driving (completing 95% of the track efficiently).

## Documentation

- [learn.md](learn.md) - In-depth explanation of RL concepts, PPO, GAE, and the codebase (Hungarian)
- [review.md](review.md) - Code review and improvement recommendations

## License

MIT License

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the CarRacing-v3 environment
- [PyTorch](https://pytorch.org/) for the deep learning framework
- PPO algorithm by [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
