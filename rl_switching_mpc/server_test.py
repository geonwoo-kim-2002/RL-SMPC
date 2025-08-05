from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from pred_msgs.msg import DetectionArray
from pred_msgs.msg import Detection
from rl_switching_mpc_srv.srv import PredOppTraj


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(PredOppTraj, 'pred_op_trajectory', self.pred_op_trajectory_callback)

    def pred_op_trajectory_callback(self, request, response):
        response.pred_opp_traj = DetectionArray()
        detect = Detection()
        detect.dt = request.ego_odom.pose.pose.position.x
        response.pred_opp_traj.detections.append(detect)
        self.get_logger().info('Incoming request\negodom: %f' % (request.ego_odom.pose.pose.position.x))
        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()