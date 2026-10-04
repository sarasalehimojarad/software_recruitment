import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from datetime import datetime
# Define the TemperatureLogger node
class TemperatureLogger(Node):

    def __init__(self, filename):
        super().__init__('temperature_logger')

        self.filename = filename

        self.subscription = self.create_subscription(
            Float32,
            '/temperature',
            self.callback,
            10
        )

    def callback(self, msg):
        if msg.data >= 50.0:
           timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           log_message = f'{timestamp} - Temperature: {msg.data:.2f} °C'

           self.get_logger().warning(log_message)

           with open(self.filename, 'a') as file:
               file.write(log_message + '\n')

def main(args=None):
    rclpy.init(args=args)

    logger = TemperatureLogger("log.txt")

    rclpy.spin(logger)

    logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



