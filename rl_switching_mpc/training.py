import gymnasium as gym
from gymnasium import spaces
from f1tenth_gym.envs.f110_env import F110Env, Track

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from pred_msgs.msg import Detection, DetectionArray
from rl_switching_mpc_srv.srv import PredOppTraj, Drive
from ament_index_python.packages import get_package_share_directory

import cv2
import time
import random
import pathlib
import numpy as np
import pandas as pd
from math import *
from transforms3d import euler
from moviepy import ImageSequenceClip
from rl_switching_mpc.Spline import Spline2D

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

import csv
import os
def log_to_csv(x, y, mode, filename="log.csv"):
    """
    x, y, mode 값을 받아서 CSV 파일에 한 줄씩 기록하는 함수
    :param x: X 좌표 (숫자 또는 문자열)
    :param y: Y 좌표 (숫자 또는 문자열)
    :param mode: 모드 값 (문자열)
    :param filename: 저장할 CSV 파일 이름 (기본값: log.csv)
    """
    file_exists = os.path.isfile(filename)

    # 파일이 없으면 헤더 추가
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["x", "y", "mode"])  # 헤더 작성
        writer.writerow([x, y, mode])  # 데이터 작성

class PredOppTrajClient(Node):
    def __init__(self):
        super().__init__("pred_opp_traj_client")
        self.scan_fov = 4.7
        self.scan_beams = 1080
        self.cli = self.create_client(PredOppTraj, 'pred_opp_trajectory')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pred_opp_traj service not available, waiting again...')
        self.req = PredOppTraj.Request()

    def send_request(self, obs, reset_collections=False):
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

        self.req.reset_collections = reset_collections
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

    def send_request(self, obs, pred_opp_traj: DetectionArray=DetectionArray(), mode: int=0, reset: bool=False):
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
        self.req.reset = reset
        return self.cli.call_async(self.req)

class MyF1TenthEnv(gym.Env, Node):
    def __init__(self, loaded_map, vehicle_params, path, training, ego_index):
        gym.Env.__init__(self)
        Node.__init__(self, "my_f1tenth_env")

        self.loaded_map = loaded_map
        self.vehicle_params = vehicle_params
        self.scale = 1.0
        f1_env = gym.make(
                    "f1tenth_gym:f1tenth-v0",
                    config={
                        "map": self.loaded_map,
                        "num_agents": 2,
                        "timestep": 0.025,
                        "integrator": "rk4",
                        "control_input": ["accl", "steering_angle"],
                        "model": "st",
                        "observation_config": {"type": "original"},
                        "params": self.vehicle_params,
                        "reset_config": {"type": "map_random_static"},
                        "scale": self.scale,
                        "lidar_dist": 0.0
                    },
                    render_mode="rgb_array"
                )

        self.f1_env = f1_env
        self.track = self.f1_env.unwrapped.track
        self.track.raceline.render_waypoints(self.f1_env.unwrapped.renderer)

        center_path = pd.read_csv(f'{path}_centerline.csv')
        self.sp = Spline2D(center_path['x_m'], center_path['y_m'])
        print("track length:", self.sp.s[-1])
        self.width_info = pd.read_csv(f'{path}_width_info.csv')
        self.max_width = max(max(self.width_info['left']), max(self.width_info['right']))
        self.min_width = min(min(self.width_info['left']), min(self.width_info['right']))

        self.max_curvature = 0.0
        self.min_curvature = 1000.0
        for i in range(int(self.sp.s[-1] * 100)):
            curr_curvature = self.sp.calc_curvature(i / 100.0)
            self.max_curvature = max(self.max_curvature, abs(curr_curvature))
            self.min_curvature = min(self.min_curvature, abs(curr_curvature))

        self.horizon = 40
        obs_dim = 4 + 4 + 2 * self.horizon + 2 * self.horizon + 1 + 1 # ego state, opp state, opp traj var, track, collision, overtaking
        # obs_dim = 4 + 4 + 3 * self.horizon + 2 * self.horizon + 5 + 1 + 1 # ego state, opp state, opp traj var, track, last 5 action, collision, overtaking
        # obs_dim = 4 + 8 * self.horizon + 2 * self.horizon + 1 + 1 + 1 # ego state, opp traj, track, mpc solved, collision, overtaking
        # obs_dim = 4 + 8 * self.horizon + 2 * self.horizon + 1 + 1 + 1 + 5 # ego state, opp traj, track, mpc solved, collision, overtaking, action change flag
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32)
        self.last_action = 0
        # self.last_action = [0, 0, 0, 0, 0]
        self.action_count = 0
        # self.action_change_flag = [0, 0, 0, 0, 0]
        self.init_s = 0.
        self.lap_count = 0
        self.reset_count = 5
        self.episode_count = 0
        self.after_overtaking = 0

        self.reset_collections = True
        self.pred_opp_traj_cli = PredOppTrajClient()
        self.ego_drive_cli = DriveClient('ego_drive')
        self.opp_drive_cli = DriveClient('opp_drive')

        self.training = training
        self.ego_index = ego_index

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)

        if self.training:
            ego_index = random.randint(0, len(self.track.centerline.xs) - 1)
            opp_index = (ego_index + random.randint(8, 12)) % len(self.track.centerline.xs)
        else:
            ego_index = self.ego_index
            opp_index = self.ego_index + 10
        print('Ego index:', ego_index, 'Opp index:', opp_index, 'Max index:', len(self.track.centerline.xs))
        initial_pose = np.array([[self.track.centerline.xs[ego_index], self.track.centerline.ys[ego_index], self.track.centerline.yaws[ego_index]], [self.track.centerline.xs[opp_index], self.track.centerline.ys[opp_index], self.track.centerline.yaws[opp_index]]])
        self.obs, info = self.f1_env.reset(options={"poses": initial_pose})
        self.last_action = 0
        # self.last_action = [0, 0, 0, 0, 0]
        self.action_count = 0
        # self.action_change_flag = [0, 0, 0, 0, 0]
        self.init_s = self.sp.find_s(self.obs['poses_x'][0], self.obs['poses_y'][0])
        self.prev_s = self.init_s
        self.curr_s = self.init_s
        self.lap_count = 0

        if self.reset_count > 2:
            self.reset_collections = True
            ed_future = self.ego_drive_cli.send_request(self.obs, reset=True)
            rclpy.spin_until_future_complete(self.ego_drive_cli, ed_future)
            od_future = self.opp_drive_cli.send_request(self.obs, reset=True)
            rclpy.spin_until_future_complete(self.opp_drive_cli, od_future)
            self.reset_count = 0
        else:
            self.reset_collections = False
            ed_future = self.ego_drive_cli.send_request(self.obs, reset=False)
            rclpy.spin_until_future_complete(self.ego_drive_cli, ed_future)
            od_future = self.opp_drive_cli.send_request(self.obs, reset=False)
            rclpy.spin_until_future_complete(self.opp_drive_cli, od_future)
            self.reset_count += 1

        return self._combine_obs(self.obs, DetectionArray(), done=False), info
        # return np.concatenate([self._combine_obs(self.obs, DetectionArray(), done=False, mpc_solved=False), np.array(self.action_change_flag, dtype=np.float32)]), info

    def step(self, action_value):
        # print("action:", action)
        action = int(action_value)
        # action = 1
        # action_value = float(action_value[0])
        # if 0.0 <= action_value < 1.0:
        #     action = 0
        # elif 1.0 <= action_value < 2.0:
        #     action = 1
        # else:
        #     action = 2

        pred_opp_traj = self._get_pred_opp_traj(self.obs, self.reset_collections)
        self.reset_collections = False

        if self.training:
            while len(pred_opp_traj.detections) != self.horizon:
                control, mpc_solved = self._get_control(0, pred_opp_traj, self.obs)
                self.obs, step_reward, done, truncated, info = self.f1_env.step(control)
                if self.after_overtaking > 0:
                    self.after_overtaking += 1
                if self.after_overtaking > 40:
                    self.after_overtaking = 0
                    done = True
                pred_opp_traj = self._get_pred_opp_traj(self.obs)
                if done:
                    print("Observation:", self.obs)
                    print("step_reward:", step_reward)
                    print("done:", done)
                    print("truncated:", truncated)
                    print("info:", info)
                    print('collision while solo driving, resetting...')
                    self.f1_env = gym.make(
                        "f1tenth_gym:f1tenth-v0",
                        config={
                            "map": self.loaded_map,
                            "num_agents": 2,
                            "timestep": 0.025,
                            "integrator": "rk4",
                            "control_input": ["accl", "steering_angle"],
                            "model": "st",
                            "observation_config": {"type": "original"},
                            "params": self.vehicle_params,
                            "reset_config": {"type": "map_random_static"},
                            "scale": self.scale,
                            "lidar_dist": 0.0
                        },
                        render_mode="rgb_array"
                    )
                    self.reset()
                    return self._combine_obs(self.obs, DetectionArray(), done=False), 0., False, True, info

        if len(pred_opp_traj.detections) != self.horizon:
            action = 0

        control, mpc_solved = self._get_control(action, pred_opp_traj, self.obs)

        self.obs, _, done, truncated, info = self.f1_env.step(control)
        # if not self.training:
        #     log_to_csv(self.obs['poses_x'][0], self.obs['poses_y'][0], action, 'ego_traj.csv')
        #     log_to_csv(self.obs['poses_x'][1], self.obs['poses_y'][1], 0, 'opp_traj.csv')

        self.curr_s = self.sp.find_s(self.obs['poses_x'][0], self.obs['poses_y'][0])
        if self._check_lap_complete():
            self.lap_count += 1
        self.prev_s = self.curr_s

        obs = self._combine_obs(self.obs, pred_opp_traj, done)

        reward = 0.0
        if self.last_action != action:
            reward -= min(0.01 * self.episode_count, 1.0)
            self.last_action = action

        var_avg = 0.0
        if len(pred_opp_traj.detections) > 0:
            for i in range(len(pred_opp_traj.detections)):
                det = pred_opp_traj.detections[i]
                var_avg += det.x_var
                var_avg += det.y_var
                var_avg += det.v_var
            var_avg /= (3 * len(pred_opp_traj.detections))

        if len(pred_opp_traj.detections) == 0:
            ego_v = self.obs['linear_vels_x'][0]
            opp_v = self.obs['linear_vels_x'][1]
            e_v = ego_v - opp_v
        else:
            ego_v = self.obs['linear_vels_x'][0]
            opp_v = pred_opp_traj.detections[0].v
            e_v = ego_v - opp_v

        var_avg = max(min(var_avg, 0.3), 0.15)
        e_v = max(min(e_v, 3.0), 0.0)
        reward += (exp(-pow(abs(var_avg - 0.15) / 0.15 * 3.0 - abs(-e_v + 3), 2) / (2 * pow(1, 2)))) * 2 - 1

        if obs[-2] == 1.0:  # collision
            # print('Collision!')
            reward -= 50.0
        elif obs[-1] == 1.0:  # success to overtake
            # print('Overtaking success!')
            reward += (1 - (var_avg - 0.15) / 0.15) * 2.0
            # done = True
            if self.after_overtaking == 0:
                self.after_overtaking = 1

        if self.after_overtaking > 0:
            if obs[-1] == 1.0 or len(pred_opp_traj.detections) != self.horizon:
                self.after_overtaking += 1
            else:
                self.after_overtaking = 0

        if self.lap_count >= 4:
            print('Reached 4 laps')
            truncated = True

        if self.after_overtaking > 40:
            self.after_overtaking = 0
            # done = True

        if done:
            self.episode_count += 1
            print(self.obs)

        if done or truncated:
            if obs[-2] == 1.0:
                print('Collision!')
            elif obs[-1] == 1.0:
                print('Overtaking success!')
        # print(action_value, 'action:', action, 'reward:', reward, 'lap count:', self.lap_count, 'ego s:', self.curr_s, 'opp s:', opp_s)
        # reward = max(reward, -10.0)
        # reward = (reward + 3.5) / 6.5
        # print('action:', action_value, 'reward:', real_reward, 'lap count:', self.lap_count)
        # print(f'action: {action_value:.1f}, reward: {reward:.4f}, lap count: {self.lap_count}, var: {var_avg:.4f}, e_v: {e_v:.4f}, episode: {self.episode_count}')

        info['action'] = action
        return obs, reward, done, truncated, info
        # return np.concatenate([obs, np.array(self.action_change_flag, dtype=np.float32)]), reward, done, truncated, info

    # def number_of_last_action(self, action):
    #     num = 0
    #     for i in range(len(self.last_action)):
    #         if self.last_action[i] == action:
    #             num += 1
    #     return num

    def _check_lap_complete(self):
        if self.prev_s - self.curr_s > self.sp.s[-1] / 2:
            go_front = True
        elif self.curr_s - self.prev_s > 0 and self.curr_s - self.prev_s < self.sp.s[-1] / 5:
            go_front = True
        else:
            go_front = False
        if go_front:
            if self.init_s <= 1e-1 and self.curr_s - self.prev_s < -self.sp.s[-1] / 2:
                return True
            if self.init_s > self.sp.s[-1] - 1e-1 and self.curr_s - self.prev_s < -self.sp.s[-1] / 2:
                return True
            if self.init_s > 1e-1 and self.init_s <= self.sp.s[-1] - 1e-1 and self.prev_s < self.init_s and self.curr_s >= self.init_s:
                return True
        return False

    def _get_pred_opp_traj(self, obs, reset_collections=False):
        curr_time = time.time()
        p_future = self.pred_opp_traj_cli.send_request(obs, reset_collections=reset_collections)
        rclpy.spin_until_future_complete(self.pred_opp_traj_cli, p_future)
        pred_opp_traj = p_future.result().pred_opp_traj
        self.pred_opp_traj = pred_opp_traj
        # print(f'Get Pred Opp Trajectory in {time.time() - curr_time:.3f} seconds')
        return pred_opp_traj

    def _get_control(self, mode, pred_opp_traj, obs):
        curr_time = time.time()
        ed_future = self.ego_drive_cli.send_request(obs, pred_opp_traj, mode)
        rclpy.spin_until_future_complete(self.ego_drive_cli, ed_future)
        ego_response = ed_future.result()
        ego_drive = ego_response.ackermann_drive
        # self.ego_drive_cli.get_logger().info('Get Ego Control input')
        self.ego_mpc_x = ego_response.mpc_x
        self.ego_mpc_y = ego_response.mpc_y
        self.ego_mpc_yaw = ego_response.mpc_yaw
        self.ego_mpc_v = ego_response.mpc_v
        # print(f'Get Ego Control input in {time.time() - curr_time:.3f} seconds')

        curr_time = time.time()
        od_future = self.opp_drive_cli.send_request(obs)
        rclpy.spin_until_future_complete(self.opp_drive_cli, od_future)
        opp_response = od_future.result()
        opp_drive = opp_response.ackermann_drive
        # self.opp_drive_cli.get_logger().info('Get Opp Control input')
        # print(f'Get Opp Control input in {time.time() - curr_time:.3f} seconds')

        return np.array([[ego_drive.drive.steering_angle, ego_drive.drive.acceleration], [opp_drive.drive.steering_angle, opp_drive.drive.acceleration]]), ego_response.is_solved

    def _combine_obs(self, obs, pred_opp_traj, done):
        # ego state
        ego_s = self.sp.find_s(obs['poses_x'][0], obs['poses_y'][0])
        ego_d = self.sp.calc_d(obs['poses_x'][0], obs['poses_y'][0], ego_s)
        ego_yaw = self.sp.calc_yaw(ego_s) - obs['poses_theta'][0]
        ego_state = np.array([0.0, (ego_d + self.max_width) / (2 * self.max_width), (ego_yaw + np.pi) / (2 * np.pi), obs['linear_vels_x'][0] / 10.0])

        # opp state
        if len(pred_opp_traj.detections) == 0:
            opp_s = self.sp.find_s(obs['poses_x'][1], obs['poses_y'][1])
            opp_d = self.sp.calc_d(obs['poses_x'][1], obs['poses_y'][1], opp_s)
            opp_yaw = self.sp.calc_yaw(opp_s) - obs['poses_theta'][1]
            opp_state = np.array([(opp_s - ego_s) / 7.0, (opp_d + self.max_width) / (2 * self.max_width), (opp_yaw + np.pi) / (2 * np.pi), obs['linear_vels_x'][1] / 10.0])
        else:
            opp_s = self.sp.find_s(pred_opp_traj.detections[0].x, pred_opp_traj.detections[0].y)
            opp_d = self.sp.calc_d(pred_opp_traj.detections[0].x, pred_opp_traj.detections[0].y, opp_s)
            opp_yaw = self.sp.calc_yaw(opp_s) - pred_opp_traj.detections[0].yaw
            if opp_s - ego_s < -self.sp.s[-1] / 2:
                opp_s += self.sp.s[-1]
            elif opp_s - ego_s > self.sp.s[-1] / 2:
                opp_s -= self.sp.s[-1]
            opp_state = np.array([(opp_s - ego_s) / 7.0, (opp_d + self.max_width) / (2 * self.max_width), (opp_yaw + np.pi) / (2 * np.pi), pred_opp_traj.detections[0].v / 10.0])

        # opp traj
        opp_traj = np.zeros((self.horizon, 2))
        # track info
        track_info = np.zeros((self.horizon, 2))
        for i in range(len(pred_opp_traj.detections)):
            det = pred_opp_traj.detections[i]
            opp_s = self.sp.find_s(det.x, det.y)

            opp_traj[i, 0] = (det.x_var + det.y_var) / 2
            opp_traj[i, 1] = det.v_var

            s = opp_s
            if s > self.sp.s[-1]:
                s -= self.sp.s[-1]
            elif s < 0.0:
                s += self.sp.s[-1]
            width_idx = round(s * 100)
            left_width = self.width_info['left'][width_idx]
            right_width = self.width_info['right'][width_idx]
            track_info[i, 0] = min(left_width, right_width) / 3.0
            # track_info[i, 0] = left_width
            # track_info[i, 1] = right_width
            track_info[i, 1] = abs(self.sp.calc_curvature(s))

        # collision, overtaking
        collision = done
        overtaking = False
        opp_curr_s = self.sp.find_s(obs['poses_x'][1], obs['poses_y'][1])
        if opp_curr_s - ego_s < -self.sp.s[-1] / 2:
            opp_curr_s += self.sp.s[-1]
        elif opp_curr_s - ego_s > self.sp.s[-1] / 2:
            opp_curr_s -= self.sp.s[-1]
        if opp_curr_s < ego_s:
            overtaking = True

        # print('ego:', len(ego_state), 'opp:', len(opp_state), 'opp_traj:', len(opp_traj.flatten()), 'track:', len(track_info.flatten()), 'collision:', collision, 'overtaking:', overtaking)
        state_vector = np.concatenate([
            ego_state.flatten(),
            opp_state.flatten(),
            opp_traj.flatten(),
            track_info.flatten(),
            np.array([float(collision), float(overtaking)], dtype=np.float32)
        ])

        # print('ego state:', ego_state)
        # print('opp state:', opp_state)
        # print('opp_traj:', opp_traj)
        # print('track_info:', track_info)

        return state_vector

    def render(self):
        return self.f1_env.render()

class RewardLoggingCallback(BaseCallback):
    def __init__(self, save_path="reward_history", rl_name='', verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.rl_name = rl_name
        self.episode_rewards = []
        self.rewards = []
        self.current_rewards = 0.0
        self.step_count = 0

    def _on_step(self) -> bool:
        # 현재 스텝의 reward 저장
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")

        if reward is not None:
            self.rewards.append(reward[0])
            self.current_rewards += reward[0]
            self.step_count += 1

        print("done:", done, ", reward:", reward)
        if done is not None and done[0] and self.step_count > 20:
            # 에피소드 종료 → 현재 reward 기록
            self.episode_rewards.append(self.current_rewards / self.step_count if self.step_count > 0 else 0.0)
            # if self.verbose > 0:
            print(f"Episode {len(self.episode_rewards)} reward: {self.episode_rewards[-1]}")

            self.current_rewards = 0.0
            self.step_count = 0

        return True  # 계속 학습

    def _on_training_end(self) -> None:
        # 학습 종료 시 저장
        curr_time = time.time()
        episode_rewards_array = np.array(self.episode_rewards)
        name = self.save_path + '_episode_' + self.rl_name + '_' + str(curr_time) + '.npy'
        np.save(name, episode_rewards_array)

        rewards_array = np.array(self.rewards)
        name = self.save_path + '_' + self.rl_name + '_' + str(curr_time) + '.npy'
        np.save(name, rewards_array)
        # if self.verbose > 0:
        print(f"Saved rewards to {name}, episode {len(self.episode_rewards)}, mean {np.mean(self.episode_rewards) if self.episode_rewards else 0.0}")
        self.rewards = []
        self.episode_rewards = []

def get_rule_based_action(obs):
    dis = obs[4] * 7.0
    avg_var = np.sum(obs[8:88]) / 80.0

    if dis < 5.0 and avg_var >= 0.3:
        action = 1
    elif dis < 5.0:
        action = 2
    else:
        action = 0
    return action

def main():
    rclpy.init()

    vehicle_params = F110Env.f1tenth_vehicle_params()
    scale = 1.0
    # path = '/home/a/rl_switching_mpc/src/RL-SMPC/maps/icra2025/icra2025'
    map_name = 'icra2025'
    pkg_path = get_package_share_directory('rl_switching_mpc')
    path = f'{pkg_path}/maps/{map_name}/{map_name}'
    map_yaml = f'{path}.yaml'
    print('Loading map from path: %s' % (map_yaml))
    map_yaml = pathlib.Path(map_yaml)
    loaded_map = Track.from_track_path(map_yaml, scale)

    rl_name = 'mpcc'
    training = False
    episode = 0
    ego_index = 10

    env = MyF1TenthEnv(loaded_map, vehicle_params, path, training, ego_index)

    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix=rl_name + '_f1tenth')
    # eval_callback = EvalCallback(env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=5000)
    reward_callback = RewardLoggingCallback(save_path="models/rewards", rl_name=rl_name, verbose=1)

    if training:
        if rl_name == 'dqn':
            # model = DQN("MlpPolicy", env, verbose=1)
            model = DQN.load("models/dqn_f1tenth_model60", env=env)
        elif rl_name == 'ppo':
            model = PPO("MlpPolicy", env, verbose=1)
            # model = PPO.load("models/re_ppo_f1tenth_model3", env=env)
        elif rl_name == 're_ppo':
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
            # model = RecurrentPPO.load("models/re_ppo_f1tenth_model5", env=env)
        elif rl_name == 'sac':
            model = SAC("MlpPolicy", env, verbose=1)
            # model = SAC.load("models/sac_f1tenth_model1", env=env)

        # while episode <= 200:
        while True:
            # model.learn(total_timesteps=4096, callback=[checkpoint_callback, eval_callback, reward_callback])
            model.learn(total_timesteps=4096, callback=[reward_callback])
            model.save("models/" + rl_name + "_f1tenth_model" + str(episode))
            print("Model saved")
            episode += 1

    if rl_name == 'dqn':
        model = DQN.load("models/dqn_f1tenth_model18")
    elif rl_name == 'ppo':
        model = PPO.load("models/ppo_f1tenth_model98")
        # model = PPO.load("models_good_col=-10/ppo_f1tenth_model234")
    elif rl_name == 're_ppo':
        model = RecurrentPPO.load("models/re_ppo_f1tenth_model199")
    elif rl_name == 'sac':
        model = SAC.load("models/sac_f1tenth_model51")

    # obs, info = env.reset()
    for ego_index in range(111):
        env.ego_index = ego_index
        obs, info = env.reset()
        frames = [env.render().copy()]
        done = False
        truncated = False
        while not (done or truncated):
            if rl_name == 'rule':
                action = get_rule_based_action(obs)
            elif rl_name == 'mpcc':
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            action = info['action']
            if 0.0 <= action < 1.0:
                action = 0
            elif 1.0 <= action < 2.0:
                action = 1
            else:
                action = 2

            log_to_csv(env.obs['poses_x'][0], env.obs['poses_y'][0], action, 'results/MPCC/trajs/' + rl_name + '_' + str(ego_index) + '_ego.csv')
            log_to_csv(env.obs['poses_x'][1], env.obs['poses_y'][1], 0, 'results/MPCC/trajs/' + rl_name + '_' + str(ego_index) + '_opp.csv')

            frame = env.render().copy()
            x_resolution, y_resolution = 0.046, 0.063
            for i in range(len(env.ego_mpc_x)):
                x_pix = int(frame.shape[1] / 2 + (env.ego_mpc_x[i] - env.obs['poses_x'][0]) / x_resolution)
                y_pix = int(frame.shape[0] / 2 - (env.ego_mpc_y[i] - env.obs['poses_y'][0]) / y_resolution)
                if action == 0:
                    color = (0, 0, 255)
                elif action == 1:
                    color = (0, 255, 0)
                elif action == 2:
                    color = (255, 0, 0)
                cv2.circle(frame, (x_pix, y_pix), 2, color, -1)
            for j in range(len(env.pred_opp_traj.detections)):
                det = env.pred_opp_traj.detections[j]
                x_pix = int(frame.shape[1] / 2 + (det.x - env.obs['poses_x'][0]) / x_resolution)
                y_pix = int(frame.shape[0] / 2 - (det.y - env.obs['poses_y'][0]) / y_resolution)
                cv2.ellipse(frame, center=(x_pix, y_pix), axes=(int(det.x_var / x_resolution), int(det.y_var / y_resolution)), angle=0, startAngle=0, endAngle=360, color=(200, 200, 200), thickness=-1)
                cv2.circle(frame, (x_pix, y_pix), 2, (255, 255, 0), -1)
            frames.append(frame)
            # if done:
            #     print("Done")
            #     clip = ImageSequenceClip(frames, fps=80)
            #     clip.write_videofile('videos/test_' + rl_name + '.mp4', codec='libx264', audio=False)
            #     obs, info = env.reset()
            #     frames = [env.render().copy()]

        clip = ImageSequenceClip(frames, fps=60)
        clip.write_videofile('results/MPCC/videos/' + rl_name + '_' + str(ego_index) + '.mp4', codec='libx264', audio=False)
    env.pred_opp_traj_cli.destroy_node()
    env.ego_drive_cli.destroy_node()
    env.opp_drive_cli.destroy_node()
    rclpy.shutdown()