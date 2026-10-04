import os
import numpy as np
import matplotlib.pyplot as plt

def generate_ablation_charts():
    # Final data from your evaluation matrix
    labels = ['PPO', 'SAC', 'TD3 (Proposed)']
    
    mean_rewards = [-525.85, 503.00, 249.99]
    mean_speeds = [64.20, 11.41, 11.03]
    collision_rates = [90.0, 100.0, 0.0]

    x = np.arange(len(labels))
    width = 0.5

    # Create a 1x3 subplot layout for the IEEE single-column format
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # 1. Collision Rate Chart (Most Important)
    bars_col = axs[0].bar(x, collision_rates, width, color=['#e74c3c', '#e74c3c', '#2ecc71'], edgecolor='black')
    axs[0].set_title('Collision Rate (%)', fontweight='bold')
    axs[0].set_ylabel('Percentage')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_ylim(0, 110)
    for bar in bars_col:
        axs[0].annotate(f'{bar.get_height()}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 2. Mean Speed Chart
    bars_spd = axs[1].bar(x, mean_speeds, width, color=['#f39c12', '#3498db', '#3498db'], edgecolor='black')
    axs[1].set_title('Mean Velocity (m/s)', fontweight='bold')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    # Draw a line representing the target safe cruising speed
    axs[1].axhline(y=15.0, color='black', linestyle='--', alpha=0.7, label='Target Speed (15 m/s)')
    axs[1].legend(fontsize=8)
    for bar in bars_spd:
        axs[1].annotate(f'{bar.get_height()}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 3. Mean Reward Chart
    bars_rew = axs[2].bar(x, mean_rewards, width, color=['#95a5a6', '#9b59b6', '#2ecc71'], edgecolor='black')
    axs[2].set_title('Mean Episode Reward', fontweight='bold')
    axs[2].set_ylabel('Cumulative Reward')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels)
    axs[2].axhline(y=0, color='black', linewidth=0.8)
    for bar in bars_rew:
        yval = bar.get_height()
        offset = 3 if yval >= 0 else -12
        axs[2].annotate(f'{yval}',
                        xy=(bar.get_x() + bar.get_width() / 2, yval),
                        xytext=(0, offset), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    
    os.makedirs("./journal_figures/", exist_ok=True)
    save_path = "./journal_figures/fig_ablation_bars.pdf"
    plt.savefig(save_path)
    print(f"\nPublication graph successfully generated and saved to {save_path}")

if __name__ == "__main__":
    generate_ablation_charts()
