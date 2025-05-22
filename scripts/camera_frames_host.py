import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class HostTalker(Node):
    def __init__(self):
        super().__init__('host_talker_node')
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
        self.timer_period = 1.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.count = 0
        self.get_logger().info('Host talker node started. Publishing to /hello_topic...')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello from Host! Count: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    host_talker = HostTalker()
    try:
        rclpy.spin(host_talker)
    except KeyboardInterrupt:
        host_talker.get_logger().info('Host talker node stopped.')
    finally:
        host_talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 