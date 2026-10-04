import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
import torch

class SACInferenceNode(Node):
    def __init__(self):
        super().__init__('sac_inference_node')
        
        # Load the frozen PyTorch model (Bypasses all NumPy/Pickle conflicts)
        self.model = torch.jit.load("./models/traced_actor.pt")
        self.model.eval()
        
        self.publisher = self.create_publisher(Float32, '/basal_ganglia/throttle_cmd', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.current_speed = 0.0
        self.get_logger().info("PyTorch Inference Node Active. Waiting for sensor data...")

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        obs = np.zeros(19, dtype=np.float32)
        obs[4] = 1.0 
        obs[5] = self.current_speed / 14.0
        
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 3.5 
        ranges[np.isnan(ranges)] = 3.5
        
        front_ranges = np.concatenate((ranges[-20:], ranges[:20]))
        front_dist = np.min(front_ranges)
        obs[7] = front_dist / 100.0 
        
        # Convert observation to PyTorch tensor and get action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model(obs_tensor).numpy()[0]
        
        throttle_msg = Float32()
        throttle_msg.data = float(action[0])
        self.publisher.publish(throttle_msg)
        
        self.get_logger().info(f"Obs front_dist: {front_dist:.2f}m | AI Throttle: {action[0]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = SACInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
