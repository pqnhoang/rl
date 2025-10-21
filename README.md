# Arm Manipulation using TD3

Reinforcement learning for robotic manipulation using Twin Delayed Deep Deterministic Policy Gradient (TD3) on Robosuite's Door environment.

## Features

- **Environment**: Robosuite Door task with Panda robot arm
- **Algorithm**: TD3 (off-policy actor-critic)
- **Key Components**: Replay buffer, twin critics, delayed policy updates, target networks
- **Monitoring**: TensorBoard logging for training visualization

## Installation

Install dependencies using the provided requirements file:

```bash
# Using conda (recommended)
conda create --name arm-manipulation --file requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Quick Start

Train the TD3 agent on the Door manipulation task:

```bash
cd model
python main.py
```

The agent will:
- Train for 10,000 episodes
- Save models every 10 episodes to `tmp/td3/`
- Log training metrics to TensorBoard in `runs/logs/`

View training progress:
```bash
tensorboard --logdir=runs/logs
```

## Project Structure

```
model/
├── main.py          # Training script
├── td3.py           # TD3 agent implementation
├── model.py         # Actor and Critic neural networks
├── buffer.py        # Experience replay buffer
└── env.py           # Environment setup
```

## Algorithm Overview

TD3 (Twin Delayed Deep Deterministic Policy Gradient) is an off-policy actor-critic algorithm that:
- Uses twin critics to reduce overestimation bias
- Delays policy updates to improve stability
- Employs target networks for stable learning
- Includes noise for exploration during training
