import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from pred_msgs.msg import Detection, DetectionArray
from rl_switching_mpc_srv.srv import PredOppTraj, Drive

import cv2
import time
import random
import pathlib
import numpy as np
from transforms3d import euler
from moviepy import ImageSequenceClip

import gymnasium as gym
from f1tenth_gym.envs.f110_env import F110Env, Track

class PredOppTrajClient(Node):
    def __init__(self):
        super().__init__("pred_opp_traj_client")
        self.scan_fov = 4.7
        self.scan_beams = 1080
        self.cli = self.create_client(PredOppTraj, 'pred_opp_trajectory')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pred_opp_traj service not available, waiting again...')
        self.req = PredOppTraj.Request()

    def send_request(self, obs):
        scan = LaserScan()
        scan.header.frame_id = 'ego_racecar/laser'
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.angle_min = -self.scan_fov / 2.
        scan.angle_max = self.scan_fov / 2.
        scan.angle_increment = self.scan_fov / self.scan_beams
        scan.range_min = 0.0
        scan.range_max = 30.0
        scan.ranges = [float(x) for x in list(obs['scans'][0])]
        self.req.scan = scan

        ego_odom = Odometry()
        ego_odom.pose.pose.position.x = float(obs['poses_x'][0])
        ego_odom.pose.pose.position.y = float(obs['poses_y'][0])
        ego_quat = euler.euler2quat(0., 0., float(obs['poses_theta'][0]), axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = float(obs['linear_vels_x'][0])
        ego_odom.twist.twist.linear.y = float(obs['linear_vels_y'][0])
        ego_odom.twist.twist.angular.z = float(obs['ang_vels_z'][0])
        self.req.ego_odom = ego_odom

        opp_odom = Odometry()
        opp_odom.pose.pose.position.x = float(obs['poses_x'][1])
        opp_odom.pose.pose.position.y = float(obs['poses_y'][1])
        opp_quat = euler.euler2quat(0., 0., float(obs['poses_theta'][1]), axes='sxyz')
        opp_odom.pose.pose.orientation.x = opp_quat[1]
        opp_odom.pose.pose.orientation.y = opp_quat[2]
        opp_odom.pose.pose.orientation.z = opp_quat[3]
        opp_odom.pose.pose.orientation.w = opp_quat[0]
        opp_odom.twist.twist.linear.x = float(obs['linear_vels_x'][1])
        opp_odom.twist.twist.linear.y = float(obs['linear_vels_y'][1])
        opp_odom.twist.twist.angular.z = float(obs['ang_vels_z'][1])
        self.req.opp_odom = opp_odom

        return self.cli.call_async(self.req)

class DriveClient(Node):
    def __init__(self, name):
        super().__init__(f"{name}_client")
        if name == 'ego_drive':
            self.index = 0
        else:
            self.index = 1
        self.cli = self.create_client(Drive, name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{name} service not available, waiting again...')
        self.req = Drive.Request()

    def send_request(self, obs, pred_opp_traj: DetectionArray=DetectionArray(), mode: int=0):
        odom = Odometry()
        odom.pose.pose.position.x = float(obs['poses_x'][self.index])
        odom.pose.pose.position.y = float(obs['poses_y'][self.index])
        quat = euler.euler2quat(0., 0., float(obs['poses_theta'][self.index]), axes='sxyz')
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        odom.pose.pose.orientation.w = quat[0]
        odom.twist.twist.linear.x = float(obs['linear_vels_x'][self.index])
        odom.twist.twist.linear.y = float(obs['linear_vels_y'][self.index])
        odom.twist.twist.angular.z = float(obs['ang_vels_z'][self.index])
        self.req.odom = odom
        self.req.pred_opp_traj = pred_opp_traj
        self.req.mode = mode
        return self.cli.call_async(self.req)

# 0: Solo  1: ACC  2: Overtake
def select_mode(pred_opp_traj, obs):
    if len(pred_opp_traj.detections) == 0:
        return 0
    else:
        return 1

def main():
    rclpy.init()

    pred_opp_traj_cli = PredOppTrajClient()
    ego_drive_cli = DriveClient('ego_drive')
    opp_drive_cli = DriveClient('opp_drive')
    vehicle_params = F110Env.f1tenth_vehicle_params()

    scale = 1.0
    path = '/home/a/rl_switching_mpc/src/RL-SMPC/maps/icra2025/icra2025'

    map_yaml = f'{path}.yaml'
    print('Loading map from path: %s' % (map_yaml))
    map_yaml = pathlib.Path(map_yaml)
    loaded_map = Track.from_track_path(map_yaml, scale)
    env = gym.make(
                    "f1tenth_gym:f1tenth-v0",
                    config={
                        "map": loaded_map,
                        "num_agents": 2,
                        "timestep": 0.025,
                        "integrator": "rk4",
                        "control_input": ["accl", "steering_angle"],
                        "model": "st",
                        "observation_config": {"type": "original"},
                        "params": vehicle_params,
                        "reset_config": {"type": "map_random_static"},
                        "scale": scale,
                        "lidar_dist": 0.0
                    },
                    render_mode="rgb_array"
                )
    track = env.unwrapped.track
    track.raceline.render_waypoints(env.unwrapped.renderer)

    ego_index = random.randint(0, len(track.raceline.xs) - 1)
    opp_index = (ego_index + random.randint(5, 7)) % len(track.raceline.xs)
    print('Ego index:', ego_index, 'Opp index:', opp_index)
    initial_pose = np.array([[track.raceline.xs[ego_index], track.raceline.ys[ego_index], track.raceline.yaws[ego_index]], [track.raceline.xs[opp_index], track.raceline.ys[opp_index], track.raceline.yaws[opp_index]]])
    obs, _ = env.reset(options={"poses": initial_pose})

    done = False
    laptime = 0.0
    start = time.time()
    end = 30.0
    frames = [env.render()]
    while laptime < end:
        p_future = pred_opp_traj_cli.send_request(obs)
        rclpy.spin_until_future_complete(pred_opp_traj_cli, p_future)
        pred_opp_traj = p_future.result().pred_opp_traj
        pred_opp_traj_cli.get_logger().info(f'Get Pred Opp Trajectory: {len(pred_opp_traj.detections)}')

        mode = select_mode(pred_opp_traj, obs)

        ed_future = ego_drive_cli.send_request(obs, pred_opp_traj, mode)
        rclpy.spin_until_future_complete(ego_drive_cli, ed_future)
        ego_response = ed_future.result()
        ego_drive = ego_response.ackermann_drive
        ego_drive_cli.get_logger().info('Get Ego Control input')

        od_future = opp_drive_cli.send_request(obs)
        rclpy.spin_until_future_complete(opp_drive_cli, od_future)
        opp_response = od_future.result()
        opp_drive = opp_response.ackermann_drive
        opp_drive_cli.get_logger().info('Get Opp Control input')

        obs, step_reward, done, truncated, info = env.step(np.array([[ego_drive.drive.steering_angle, ego_drive.drive.acceleration], [opp_drive.drive.steering_angle, opp_drive.drive.acceleration]]))
        laptime += step_reward

        print('laptime:', laptime, 'done:', done)
        frame = env.render().copy()
        x_resolution, y_resolution = 0.046 / 1.0, 0.063 / 1.0
        # x_pix = int(frame.shape[1] / 2 + (obs['poses_x'][1] - obs['poses_x'][0]) / x_resolution)
        # y_pix = int(frame.shape[0] / 2 - (obs['poses_y'][1] - obs['poses_y'][0]) / y_resolution)
        # cv2.circle(frame, (x_pix, y_pix), 5, (0, 255, 0), -1)
        for i in range(len(ego_response.mpc_x)):
            x_pix = int(frame.shape[1] / 2 + (ego_response.mpc_x[i] - obs['poses_x'][0]) / x_resolution)
            y_pix = int(frame.shape[0] / 2 - (ego_response.mpc_y[i] - obs['poses_y'][0]) / y_resolution)
            if mode == 0:
                color = (0, 0, 255)
            elif mode == 1:
                color = (0, 255, 0)
            elif mode == 2:
                color = (255, 0, 0)
            cv2.circle(frame, (x_pix, y_pix), 2, color, 1)
        for i in range(len(pred_opp_traj.detections)):
            det = pred_opp_traj.detections[i]
            x_pix = int(frame.shape[1] / 2 + (det.x - obs['poses_x'][0]) / x_resolution)
            y_pix = int(frame.shape[0] / 2 - (det.y - obs['poses_y'][0]) / y_resolution)
            cv2.circle(frame, (x_pix, y_pix), 2, (255, 0, 0), 1)
        frames.append(frame)
        print('Ego pose:', obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])

        if done or laptime >= end:
            clip = ImageSequenceClip(frames, fps=80)
            clip.write_videofile('videos/test_' + str(laptime) + '.mp4', codec='libx264', audio=False)
            ego_index = random.randint(0, len(track.raceline.xs) - 1)
            opp_index = (ego_index + random.randint(10, int(len(track.raceline.xs) / 4))) % len(track.raceline.xs)
            print('Ego index:', ego_index, 'Opp index:', opp_index)
            initial_pose = np.array([[track.raceline.xs[ego_index], track.raceline.ys[ego_index], track.raceline.yaws[ego_index]], [track.raceline.xs[opp_index], track.raceline.ys[opp_index], track.raceline.yaws[opp_index]]])
            obs, _ = env.reset(options={"poses": initial_pose})
            frames = [env.render()]

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)

    env.close()

    pred_opp_traj_cli.destroy_node()
    ego_drive_cli.destroy_node()
    opp_drive_cli.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
