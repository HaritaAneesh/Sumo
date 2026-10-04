import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import Observation, ActionSignal
import numpy as np
import random

class QLearningAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')
        
        # Q-Table: Stores Q-values for (state, action)
        # State: tuple of (lidar_bin, traffic_light_green_int)
        self.q_table = {}
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.95   # Discount factor
        self.epsilon = 0.2  # Exploration rate
        
        # Subscribe to Environment Observations
        self.subscription = self.create_subscription(
            Observation, '/traffic_env/observation', self.obs_callback, 10)
        
        # Publish proposed actions to the Gatekeeper
        self.publisher_ = self.create_publisher(
            ActionSignal, '/rl_agent/proposed_action', 10)
        
        self.get_logger().info('RL Agent Node Initialized')

    def discretize_state(self, obs):
        # Convert continuous LiDAR to 3 bins: 0 (Close), 1 (Mid), 2 (Far)
        min_lidar = min(obs.lidar_proximity)
        lidar_bin = 0 if min_lidar < 1.0 else 1 if min_lidar < 3.0 else 2
        return (lidar_bin, int(obs.traffic_light_green))

    def obs_callback(self, obs):
        state = self.discretize_state(obs)
        
        # Ensure state exists in Q-table
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0] # [Stop, Slow, Accel]
            
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1, 2])
        else:
            action = np.argmax(self.q_table[state])
            
        # Publish Action
        msg = ActionSignal()
        msg.steering_angle = 0.0
        msg.throttle = float(action * 0.4) 
        msg.confidence_score = 0.8
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = QLearningAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
