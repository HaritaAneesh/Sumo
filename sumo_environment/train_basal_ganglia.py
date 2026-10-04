import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoTrafficEnv

# Initialize environment
env = SumoTrafficEnv(gui=False, step_length=0.1)

# Configure the CheckpointCallback to save the model every 20,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./models/",
    name_prefix="basal_ganglia_gating",
    save_replay_buffer=True, # Allows you to pause and perfectly resume SAC training later
    save_vecnormalize=True
)

# Initialize the SAC model
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.99,
    tau=0.005,
    verbose=1,
    device="cuda",
    tensorboard_log="./sac_tensorboard/"
)

# Train the agent with the limited timestep ceiling and the callback attached
model.learn(total_timesteps=200000, callback=checkpoint_callback, log_interval=10)

# Final model save once the 200,000 steps are fully complete
model.save("models/basal_ganglia_gating_weights_final")
env.close()
