import sys

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from pred_msgs.msg import DetectionArray
from pred_msgs.msg import Detection
from rl_switching_mpc_srv.srv import PredOppTraj

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(PredOppTraj, 'pred_op_trajectory')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = PredOppTraj.Request()

    def send_request(self):
        self.req.ego_odom = Odometry()
        self.req.ego_odom.pose.pose.position.x = 12.3
        return self.cli.call_async(self.req)


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    future = minimal_client.send_request()
    rclpy.spin_until_future_complete(minimal_client, future)
    response = future.result()
    minimal_client.get_logger().info('%f' % response.pred_opp_traj.detections[0].dt)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()