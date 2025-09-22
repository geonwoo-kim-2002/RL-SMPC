import numpy as np
import matplotlib.pyplot as plt
import glob
import os

plot_episode = True
mode = 'ppo'

path = 'rewards_'
if plot_episode:
    path += 'episode_'
path += mode

# 모든 rewards*.npy 파일 찾기
files = glob.glob('models_good_col=-10/' + path + "_*.npy")
# time 값 기준으로 정렬 (파일명 뒤의 숫자 기준)
files.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0].replace(path + "_", "")))

all_rewards = []

# for f in files:
#     rewards = np.load(f, allow_pickle=True)
#     all_rewards.extend(rewards)  # 리스트로 합치기
#     print(f"Loaded {f}, length={len(rewards)}")
rewards = np.load(files[-1], allow_pickle=True)
all_rewards.extend(rewards)
print(f"Loaded {files[-1]}, length={len(rewards)}")

# numpy 배열로 변환
all_rewards = np.array(all_rewards[:2000])

# --- Sliding Window 평균 ---
window_size = 500  # 원하는 window 크기 설정
cumsum = np.cumsum(np.insert(all_rewards, 0, 0))  # 누적합
moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
episodes = np.arange(window_size, len(all_rewards) + 1)

# Plot
plt.figure(figsize=(12, 6))
# plt.plot(np.arange(len(all_rewards)), all_rewards, alpha=0.3, label='Raw Rewards')  # 원본 값 (투명하게)
plt.plot(episodes, moving_avg, color='red', label=f'PPO')




plot_episode = True
mode = 'dqn'

path = 'rewards_'
if plot_episode:
    path += 'episode_'
path += mode

# 모든 rewards*.npy 파일 찾기
files = glob.glob('models_good_col=-10/' + path + "_*.npy")
# time 값 기준으로 정렬 (파일명 뒤의 숫자 기준)
files.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0].replace(path + "_", "")))

all_rewards = []

for f in files:
    rewards = np.load(f, allow_pickle=True)
    all_rewards.extend(rewards)  # 리스트로 합치기
    print(f"Loaded {f}, length={len(rewards)}")
# rewards = np.load(files[-1], allow_pickle=True)
# all_rewards.extend(rewards)
# print(f"Loaded {files[-1]}, length={len(rewards)}")

# numpy 배열로 변환
all_rewards = np.array(all_rewards[:2000])

# --- Sliding Window 평균 ---
# window_size = 500  # 원하는 window 크기 설정
cumsum = np.cumsum(np.insert(all_rewards, 0, 0))  # 누적합
moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
episodes = np.arange(window_size, len(all_rewards) + 1)

# Plot
# plt.plot(np.arange(len(all_rewards)), all_rewards, alpha=0.3, label='Raw Rewards')  # 원본 값 (투명하게)
plt.plot(episodes, moving_avg, label=f'DQN')




plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes (Sliding Window)')
plt.legend()
plt.grid(True)
plt.show()