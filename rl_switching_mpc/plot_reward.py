import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 모든 rewards*.npy 파일 찾기
files = glob.glob("models/rewards*.npy")

# time 값 기준으로 정렬 (파일명 뒤의 숫자 기준)
files.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0].replace("rewards", "")))

all_rewards = []

for f in files:
    rewards = np.load(f, allow_pickle=True)
    all_rewards.extend(rewards)  # 리스트로 합치기
    print(f"Loaded {f}, length={len(rewards)}")

# numpy 배열로 변환
all_rewards = np.array(all_rewards)
# all_rewards = all_rewards[30000:]
# Plot
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(all_rewards)), all_rewards, label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes (Merged)')
plt.legend()
plt.grid(True)
plt.show()