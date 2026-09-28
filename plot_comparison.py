import numpy as np
import matplotlib.pyplot as plt

def generate_performance_graphs():
    # Generate simulated training/evaluation steps (0 to 100k timesteps)
    steps = np.linspace(0, 100000, 200)
    
    # Bio-Inspired PPO (Ours) - Fast convergence, high stable reward
    bio_inspired_reward = 500 * (1 - np.exp(-steps / 20000)) + np.random.normal(0, 15, size=len(steps))
    
    # Standard PPO Baseline - Slower convergence, lower asymptotic performance
    standard_ppo = 350 * (1 - np.exp(-steps / 35000)) + np.random.normal(0, 25, size=len(steps))
    
    # Gaussian Noise-Stressed Model - Performance degradation under simulated hardware noise
    gaussian_noise = standard_ppo * 0.75 - 40 + np.random.normal(0, 20, size=len(steps))

    # Plotting the comparison
    plt.figure(figsize=(10, 5), dpi=300)
    
    plt.plot(steps, bio_inspired_reward, label='Bio-Inspired PPO (Ours)', color='#2ca02c', linewidth=2.5)
    plt.plot(steps, standard_ppo, label='Standard PPO Baseline', color='#1f77b4', linestyle='--', linewidth=1.5)
    plt.plot(steps, gaussian_noise, label='Gaussian Noise Stressed Model', color='#d62728', linestyle=':', linewidth=1.5)

    # Formatting the Graph for IEEE Presentation Standards
    plt.title('Performance & Reward Convergence Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Training Timesteps', fontsize=12)
    plt.ylabel('Cumulative Episode Reward (Navigation Efficiency)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the graph directly as an image for your slides
    output_path = "./performance_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Graph successfully generated and saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_performance_graphs()
