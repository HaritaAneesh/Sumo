import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import Observation, ActionSignal
import random

class TrafficEnvironmentNode(Node):
    def __init__(self):
        super().__init__('traffic_env_node')
        self.obs_pub = self.create_publisher(Observation, '/traffic_env/observation', 10)
        self.sub = self.create_subscription(ActionSignal, '/rl_agent/proposed_action', self.action_callback, 10)
        self.timer = self.create_timer(0.5, self.publish_observation)
        
    def publish_observation(self):
        obs = Observation()
        # Simulated environment state
        obs.velocity = 5.0
        obs.distance_to_intersection = 10.0
        obs.lidar_proximity = [random.uniform(0, 5), random.uniform(0, 5), random.uniform(0, 5)]
        obs.traffic_light_green = True
        self.obs_pub.publish(obs)

    def action_callback(self, msg):
        self.get_logger().info(f'Env received action: Throttle={msg.throttle}')

def main(args=None):
    rclpy.init(args=args)
    node = TrafficEnvironmentNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
