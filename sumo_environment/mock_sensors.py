import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class MockTurtleBot(Node):
    def __init__(self):
        super().__init__('mock_turtlebot')
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(1.0, self.publish_fake_data)
        self.get_logger().info("Mock TurtleBot Active. Broadcasting fake sensor data...")

    def publish_fake_data(self):
        # Mock LiDAR: 360 degrees of clear space (3.5m maximum range)
        scan = LaserScan()
        scan.ranges = [3.5] * 360
        
        # Inject a fake obstacle just 0.8 meters directly in front of the robot
        scan.ranges[0] = 0.8 
        self.scan_pub.publish(scan)

        # Mock Odometry: Pretend the vehicle is currently rolling forward at 0.3 m/s
        odom = Odometry()
        odom.twist.twist.linear.x = 0.3
        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = MockTurtleBot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
