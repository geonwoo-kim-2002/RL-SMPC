import numpy as np
import gymnasium as gym
from gymnasium import spaces
from f1tenth_gym.envs.f110_env import F110Env, Track

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from pred_msgs.msg import Detection, DetectionArray
from rl_switching_mpc_srv.srv import PredOppTraj, Drive
from transforms3d import euler
import random
from rl_switching_mpc.Spline import Spline, Spline2D
import pandas as pd
from stable_baselines3 import PPO
import pathlib
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import cv2
from moviepy import ImageSequenceClip
import time
from stable_baselines3.common.callbacks import BaseCallback

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
    def __init__(self, f1tenth_env, path):
        gym.Env.__init__(self)
        Node.__init__(self, "my_f1tenth_env")

        self.f1_env = f1tenth_env
        self.track = self.f1_env.unwrapped.track
        self.track.raceline.render_waypoints(self.f1_env.unwrapped.renderer)

        center_path = pd.read_csv(f'{path}_centerline.csv')
        self.sp = Spline2D(center_path['x_m'], center_path['y_m'])
        print("track length:", self.sp.s[-1])
        self.width_info = pd.read_csv(f'{path}_width_info.csv')

        self.horizon = 40
        obs_dim = 4 + 8 * self.horizon + 2 * self.horizon + 1 + 1 + 1 # ego state, opp traj, track, mpc solved, collision, overtaking
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.last_action = 0
        self.action_count = 0

        self.reset_collections = True
        self.pred_opp_traj_cli = PredOppTrajClient()
        self.ego_drive_cli = DriveClient('ego_drive')
        self.opp_drive_cli = DriveClient('opp_drive')

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)

        ego_index = random.randint(0, len(self.track.raceline.xs) - 1)
        opp_index = (ego_index + random.randint(8, 12)) % len(self.track.raceline.xs)
        print('Ego index:', ego_index, 'Opp index:', opp_index)
        initial_pose = np.array([[self.track.raceline.xs[ego_index], self.track.raceline.ys[ego_index], self.track.raceline.yaws[ego_index]], [self.track.raceline.xs[opp_index], self.track.raceline.ys[opp_index], self.track.raceline.yaws[opp_index]]])
        self.obs, info = self.f1_env.reset(options={"poses": initial_pose})
        self.reset_collections = True
        self.last_action = 0
        self.action_count = 0
        ed_future = self.ego_drive_cli.send_request(self.obs, reset=True)
        rclpy.spin_until_future_complete(self.ego_drive_cli, ed_future)
        od_future = self.opp_drive_cli.send_request(self.obs, reset=True)
        rclpy.spin_until_future_complete(self.opp_drive_cli, od_future)
        return self._combine_obs(self.obs, DetectionArray(), done=False, mpc_solved=False), info

    def step(self, action):
        # print("action:", action)
        action = int(action)

        pred_opp_traj = self._get_pred_opp_traj(self.obs, self.reset_collections)
        self.reset_collections = False

        while len(pred_opp_traj.detections) != self.horizon:
            control, mpc_solved = self._get_control(0, pred_opp_traj, self.obs)
            self.obs, step_reward, done, truncated, info = self.f1_env.step(control)
            pred_opp_traj = self._get_pred_opp_traj(self.obs)
            if done:
                print("Observation:", self.obs, flush=True)
                print("step_reward:", step_reward, flush=True)
                print("done:", done, flush=True)
                print("truncated:", truncated, flush=True)
                print("info:", info, flush=True)
                print('collision while solo driving, resetting...', flush=True)
                # return self._combine_obs(self.obs, DetectionArray(), done=False, mpc_solved=True), 0., False, truncated, info
                self.reset()

        control, mpc_solved = self._get_control(action, pred_opp_traj, self.obs)
        self.obs, _, done, truncated, info = self.f1_env.step(control)

        obs = self._combine_obs(self.obs, pred_opp_traj, done, mpc_solved=mpc_solved)
        reward = action * -0.05
        if obs[-2] == 1.0:  # collision
            print('Collision!', flush=True)
            reward = -0.5 - reward
        elif obs[-1] == 1.0:  # success to overtake
            if action == 2:
                print('Overtaking success!', flush=True)
                reward = 1.
            done = True

        if self.action_count >= 1 and action != self.last_action:
            reward -= 0.2
            self.action_count = 1
            self.last_action = action
        elif action != self.last_action:
            self.action_count = 1
            self.last_action = action
        elif self.action_count == 3:
            self.action_count = 0
        elif self.action_count >= 1:
            self.action_count += 1

        if obs[-3] == 0.0:  # mpc not solved
            reward -= 0.1

        print(self.action_count, 'action:', action, 'reward:', reward)

        return obs, reward, done, truncated, info

    def _get_pred_opp_traj(self, obs, reset_collections=False):
        curr_time = time.time()
        p_future = self.pred_opp_traj_cli.send_request(obs, reset_collections=reset_collections)
        rclpy.spin_until_future_complete(self.pred_opp_traj_cli, p_future)
        pred_opp_traj = p_future.result().pred_opp_traj
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

    def _combine_obs(self, obs, pred_opp_traj, done, mpc_solved):
        # ego state
        ego_s = self.sp.find_s(obs['poses_x'][0], obs['poses_y'][0])
        ego_d = self.sp.calc_d(obs['poses_x'][0], obs['poses_y'][0], ego_s)
        ego_yaw = self.sp.calc_yaw(ego_s) - obs['poses_theta'][0]
        ego_state = np.array([0.0, ego_d, ego_yaw, obs['linear_vels_x'][0]])

        # opp traj
        opp_traj = np.zeros((self.horizon, 8))
        for i in range(len(pred_opp_traj.detections)):
            det = pred_opp_traj.detections[i]
            opp_s = self.sp.find_s(det.x, det.y)
            opp_d = self.sp.calc_d(det.x, det.y, opp_s)
            opp_yaw = self.sp.calc_yaw(opp_s) - det.yaw

            if opp_s - ego_s < -self.sp.s[-1] / 2:
                opp_s += self.sp.s[-1]
            elif opp_s - ego_s > self.sp.s[-1] / 2:
                opp_s -= self.sp.s[-1]
            opp_traj[i, 0] = opp_s - ego_s
            opp_traj[i, 1] = opp_d
            opp_traj[i, 2] = opp_yaw
            opp_traj[i, 3] = det.v

            yaw_track = self.sp.calc_yaw(opp_s)
            cos_y = np.cos(yaw_track)
            sin_y = np.sin(yaw_track)

            # 공분산 행렬 (x, y)
            cov_xy = np.array([[det.x_var, 0.0],
                               [0.0, det.y_var]])

            # 회전변환 (R * cov_xy * R^T)
            R = np.array([[cos_y, sin_y],
                          [-sin_y, cos_y]])
            cov_sd = R @ cov_xy @ R.T

            s_var = cov_sd[0, 0]
            d_var = cov_sd[1, 1]

            opp_traj[i, 4] = s_var
            opp_traj[i, 5] = d_var
            opp_traj[i, 6] = det.yaw_var
            opp_traj[i, 7] = det.v_var
            # print('s var:', s_var, 'd var:', d_var, 'yaw var:', det.yaw_var, 'v var:', det.v_var)

        # track
        track_info = np.zeros((self.horizon, 2))
        for i in range(self.horizon):
            s = ego_s + (i + 1) * 0.4
            if s > self.sp.s[-1]:
                s -= self.sp.s[-1]
            width_idx = round(s * 100)
            left_width = self.width_info['left'][width_idx]
            right_width = self.width_info['right'][width_idx]
            track_info[i, 0] = left_width + right_width
            track_info[i, 1] = self.sp.calc_curvature(s)

        # mpc_solved, collision, overtaking
        collision = done
        overtaking = False
        opp_curr_s = self.sp.find_s(obs['poses_x'][1], obs['poses_y'][1])
        if opp_curr_s - ego_s < -self.sp.s[-1] / 2:
            opp_curr_s += self.sp.s[-1]
        elif opp_curr_s - ego_s > self.sp.s[-1] / 2:
            opp_curr_s -= self.sp.s[-1]
        if opp_curr_s < ego_s:
            overtaking = True

        state_vector = np.concatenate([
            ego_state.flatten(),
            opp_traj.flatten(),
            track_info.flatten(),
            np.array([float(mpc_solved), float(collision), float(overtaking)], dtype=np.float32)
        ])

        return state_vector

    def render(self):
        return self.f1_env.render()

class RewardLoggingCallback(BaseCallback):
    def __init__(self, save_path="reward_history", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.rewards = []

    def _on_step(self) -> bool:
        # 현재 스텝의 reward 저장
        reward = self.locals["rewards"]  # VecEnv니까 np.ndarray
        self.rewards.append(reward.copy())

        return True  # 계속 학습

    def _on_training_end(self) -> None:
        # 학습 종료 시 저장
        rewards_array = np.array(self.rewards)
        np.save(self.save_path + str(time.time()) + '.npy', rewards_array)
        if self.verbose > 0:
            print(f"Saved rewards to {self.save_path + str(time.time()) + '.npy'}")

def main():
    rclpy.init()

    vehicle_params = F110Env.f1tenth_vehicle_params()
    scale = 1.0
    path = '/home/a/rl_switching_mpc/src/RL-SMPC/maps/icra2025/icra2025'

    map_yaml = f'{path}.yaml'
    print('Loading map from path: %s' % (map_yaml))
    map_yaml = pathlib.Path(map_yaml)
    loaded_map = Track.from_track_path(map_yaml, scale)
    f1_env = gym.make(
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

    env = MyF1TenthEnv(f1_env, path)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix='ppo_f1tenth')
    eval_callback = EvalCallback(env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=5000)
    reward_callback = RewardLoggingCallback(save_path="models/rewards", verbose=1)
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    # model = PPO.load("models/ppo_f1tenth_model", env=env, device='cpu')

    episode = 0
    while True:
        model.learn(total_timesteps=4096, callback=[checkpoint_callback, eval_callback, reward_callback])
        model.save("models/ppo_f1tenth_model" + str(episode))
        episode += 1

    model = PPO.load("models/ppo_f1tenth_model30")

    obs, info = env.reset()
    frames = [env.render().copy()]
    for i in range(1200):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        frame = env.render().copy()
        x_resolution, y_resolution = 0.046, 0.063
        for j in range(len(env.ego_mpc_x)):
            x_pix = int(frame.shape[1] / 2 + (env.ego_mpc_x[j] - env.obs['poses_x'][0]) / x_resolution)
            y_pix = int(frame.shape[0] / 2 - (env.ego_mpc_y[j] - env.obs['poses_y'][0]) / y_resolution)
            if action == 0:
                color = (0, 0, 255)
            elif action == 1:
                color = (0, 255, 0)
            elif action == 2:
                color = (255, 0, 0)
            cv2.circle(frame, (x_pix, y_pix), 2, color, 1)
        frames.append(frame)
        if done:
            print("Done")
            clip = ImageSequenceClip(frames, fps=80)
            clip.write_videofile('videos/test' + str(i) + '.mp4', codec='libx264', audio=False)
            obs, info = env.reset()
            frames = [env.render().copy()]

    clip = ImageSequenceClip(frames, fps=80)
    clip.write_videofile('videos/test' + str(i) + '.mp4', codec='libx264', audio=False)
    env.pred_opp_traj_cli.destroy_node()
    env.ego_drive_cli.destroy_node()
    env.opp_drive_cli.destroy_node()
    rclpy.shutdown()