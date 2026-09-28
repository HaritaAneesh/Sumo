# 🚦 Bio-Inspired Reinforcement Learning for Urban Traffic Navigation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-Stable%20Baselines3-orange)
![SUMO](https://img.shields.io/badge/Simulator-Eclipse%20SUMO-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)

> **An advanced autonomous driving simulation tackling high-density urban congestion using Deep Reinforcement Learning.**

## 📖 Project Overview
Urban traffic congestion is a complex, dynamic problem. This project simulates a highly congested traffic network using **Eclipse SUMO** (Simulation of Urban MObility) and trains an autonomous ego-vehicle to navigate it efficiently. 

Our core objective was to develop an agent capable of weaving through dense traffic (simulating environments with 1,500+ active vehicles) without causing collisions or succumbing to traffic gridlocks.

## 🚀 The Journey & What We Have Achieved

### Phase 1: The PPO Baseline & The "Safe Local Optimum" Problem
We initially implemented a **Proximal Policy Optimization (PPO)** agent. While PPO learned to avoid collisions, it quickly fell into a "safe local optimum." In heavy traffic, the PPO agent discovered that the easiest way to avoid negative rewards (crashes) was to simply stop moving entirely or drive excessively passively. This resulted in severe traffic stalling and poor navigational efficiency.

### Phase 2: The SAC Upgrade (Maximum Entropy RL)
To solve the freezing behavior, we transitioned to a **Soft Actor-Critic (SAC)** architecture. SAC incorporates **Maximum Entropy Reinforcement Learning**, which mathematically forces the agent to explore and maintain dynamic movement rather than playing it safe. 

### Phase 3: Marathon Training & Superior Results
We trained the SAC agent in a marathon simulation, passing **150,000+ steps** in an environment with over 70,000 total spawned vehicles. 
* **The Result:** The SAC agent successfully learned to aggressively yet safely weave through heavy traffic.
* **Comparative Study:** Our generated metrics prove SAC significantly outperforms PPO, climbing to a mean episode reward of ~90, compared to PPO stalling at ~25.

## 🛠️ Tech Stack & Architecture
* **Environment:** Custom SUMO Traffic wrapper integrated via `TraCI`.
* **Algorithms:** Proximal Policy Optimization (PPO) & Soft Actor-Critic (SAC).
* **Libraries:** `stable-baselines3`, `gymnasium`, `PyTorch`.
* **Data Visualization:** `matplotlib`, `numpy`.

## 📂 Repository Structure
```text
📦 Sumo
 ┣ 📂 sumo_traffic_env/      # Custom TraCI-based SUMO environment
 ┣ 📂 logs/                  # Training checkpoints & tensorboard logs (e.g., sac_ckpt_150000_steps.zip)
 ┣ 📂 evaluation_results/    # Visual graphs comparing PPO and SAC performance
 ┣ 📜 train_ppo.py           # Training pipeline for the PPO agent
 ┣ 📜 train_sac.py           # Training pipeline for the SAC agent
 ┣ 📜 evaluate_sac.py        # GUI visualizer to watch the trained SAC agent drive
 ┣ 📜 compare_ppo_sac.py     # Evaluation script to plot learning curves
 ┗ 📜 README.md              # Project documentation
