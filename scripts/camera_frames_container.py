import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ContainerListener(Node):
    def __init__(self):
        super().__init__('container_listener_node')
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Container listener node started. Subscribed to /hello_topic...')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received from host: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    container_listener = ContainerListener()
    try:
        rclpy.spin(container_listener)
    except KeyboardInterrupt:
        container_listener.get_logger().info('Container listener node stopped.')
    finally:
        container_listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 