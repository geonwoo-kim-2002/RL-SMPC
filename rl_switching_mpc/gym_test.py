import rclpy
from rclpy.node import Node

import time
import numpy as np
import cv2
from moviepy import ImageSequenceClip

import pathlib
import gymnasium as gym
from f1tenth_gym.envs.f110_env import F110Env, Track

def world_to_image(x_world, y_world, resolution=0.07, origin=(-16.3, -14.5), image_height=None):
    """
    월드 좌표 (x_world, y_world) → 이미지 픽셀 좌표 (x_pixel, y_pixel)

    - origin: 월드 좌표계에서 맵 이미지의 왼쪽 아래 기준 (m)
    - resolution: m/pixel
    - image_height: 맵 이미지 높이 (pixel) - y축 반전 위해 필요
    """
    x_pixel = int((x_world - origin[0]) / resolution)
    y_pixel = int(image_height - (y_world - origin[1]) / resolution)

    return x_pixel, y_pixel

if __name__ == "__main__":
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

    # env = gym.wrappers.RecordVideo(env, f"videos/video_{time.time()}")
    track = env.unwrapped.track
    track.raceline.render_waypoints(env.unwrapped.renderer)

    initial_pose = np.array([[track.raceline.xs[0], track.raceline.ys[0], track.raceline.yaws[0]], [track.raceline.xs[10], track.raceline.ys[10], track.raceline.yaws[10]]])
    obs, _ = env.reset(options={"poses": initial_pose})
    # obs, _, done, _, _ = env.step(np.array([[0.05, 5.0], [0.0, 2.0]]))

    done = False
    laptime = 0.0
    start = time.time()
    frames = [env.render()]
    while laptime < 5.0:
        obs, step_reward, done, truncated, info = env.step(np.array([[-0.0, 5.0], [0.0, 3.0]]))
        laptime += step_reward

        print('laptime:', laptime, 'done:', done)
        # print(obs)
        frame = env.render().copy()
        x_pix, y_pix = world_to_image(obs['poses_x'][0], obs['poses_y'][0], image_height=frame.shape[0])
        cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2) - 10), 10, (0, 255, 0), -1)
        frames.append(frame)

        if done:
            clip = ImageSequenceClip(frames, fps=60)
            clip.write_videofile('videos/test_' + str(laptime) + '.mp4', codec='libx264', audio=False)

            obs, _ = env.reset(options={"poses": initial_pose})
            frames = [env.render()]

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)

    env.close()