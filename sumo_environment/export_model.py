import torch
from stable_baselines3 import SAC

print("Loading SB3 checkpoint...")
model = SAC.load("./models/basal_ganglia_gating_20000_steps")

class DeterministicActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        return self.actor(obs, deterministic=True)

print("Tracing static PyTorch graph...")
wrapped_actor = DeterministicActor(model.policy.actor)
dummy_obs = torch.zeros(1, 19)
traced_model = torch.jit.trace(wrapped_actor, dummy_obs)

traced_model.save("./models/traced_actor.pt")
print("Success! Saved ROS-safe model to ./models/traced_actor.pt")
