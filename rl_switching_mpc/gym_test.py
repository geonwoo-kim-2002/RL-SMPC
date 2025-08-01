import rclpy
from rclpy.node import Node

import time
import numpy as np

import pathlib
import gymnasium as gym
from f1tenth_gym.envs.f110_env import F110Env, Track

if __name__ == "__main__":
    vehicle_params = F110Env.f1tenth_vehicle_params()

    scale = 1.0
    path = '/home/a/rl_switching_mpc/src/RL-SMPC/maps/icra2025/icra2025'

    map_yaml = f'{path}.yaml'
    map_centerline = f'{path}_centerline.csv'
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
                    render_mode="rgb_array",
                )

    env = gym.wrappers.RecordVideo(env, f"video_{time.time()}")
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
        # action = env.action_space.sample()
        # obs, step_reward, done, truncated, info = env.step(action)
        obs, step_reward, done, truncated, info = env.step(np.array([[-0.0, 5.0], [0.0, 3.0]]))
        laptime += step_reward

        if done:
            obs, _ = env.reset(options={"poses": initial_pose})

        frame = env.render()
        frames.append(frame)

        print('laptime:', laptime, 'done:', done)

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)

    env.close()