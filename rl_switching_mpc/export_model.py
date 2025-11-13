import torch
from stable_baselines3 import PPO

# 학습된 모델 불러오기
model = PPO.load("models_good_col=-10/ppo_f1tenth_model134", device="cpu")

# 정책 네트워크 가져오기
policy = model.policy

# 예시 입력 (환경의 관측 공간 크기와 일치해야 함)
import numpy as np
obs = np.zeros((1,) + model.observation_space.shape, dtype=np.float32)
obs_tensor = torch.as_tensor(obs)

# TorchScript 변환
traced_policy = torch.jit.trace(policy, obs_tensor)

# 저장
traced_policy.save("ppo_policy_traced.pt")
print("TorchScript 모델 저장 완료")