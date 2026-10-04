import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import ActionSignal
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class BasalGangliaGatekeeper(Node):
    def __init__(self):
        super().__init__('bg_gatekeeper')
        self.emergency_stop = False
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription = self.create_subscription(ActionSignal, '/rl_agent/proposed_action', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('BG Gatekeeper Initialized')

    def scan_callback(self, msg):
        if min(msg.ranges) < 0.5:
            self.emergency_stop = True
        else:
            self.emergency_stop = False

    def listener_callback(self, msg):
        if self.emergency_stop:
            self.publisher_.publish(Twist())
            return
        if msg.confidence_score < 0.5:
            self.publisher_.publish(Twist())
        else:
            cmd = Twist()
            cmd.linear.x = float(msg.throttle)
            cmd.angular.z = float(msg.steering_angle)
            self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = BasalGangliaGatekeeper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
