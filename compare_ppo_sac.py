import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory for results if it doesn't exist
os.makedirs('./evaluation_results', exist_ok=True)

# Example data structure for comparison
# (If you logged your evaluation rewards into arrays or files, load them here)
steps = [0, 50000, 100000, 150000]

# Placeholder or loaded reward metrics comparing PPO vs SAC behavior
# Replace these with your actual recorded evaluation reward lists
ppo_rewards = [-120.5, -45.2, 10.4, 25.0]  # Example PPO scores
sac_rewards = [-90.0, -15.5, 45.8, 88.2]  # Example SAC scores (showing smoother/higher reward)

plt.figure(figsize=(10, 6))
plt.plot(steps, ppo_rewards, marker='o', linestyle='-', label='PPO Agent', color='blue')
plt.plot(steps, sac_rewards, marker='s', linestyle='-', label='SAC Agent (Proposed)', color='orange')

plt.title('Performance Comparison: PPO vs. SAC in SUMO Traffic Navigation', fontsize=12, fontweight='bold')
plt.xlabel('Training Steps', fontsize=10)
plt.ylabel('Mean Episode Reward', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

output_path = './evaluation_results/ppo_vs_sac_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Comparison graph successfully saved to {output_path}")
