import torch
import torch.nn as nn
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation
import time

from utils import PPODropoutInjector

env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = GrayscaleObservation(env, keep_dim=True)

model = PPODropoutInjector(
    "./gymmodel/CarRacing.zip",
    cnn_dropout_rate=0.05,
    mlp_dropout_rate=0.05,
    env=env
)

mc_samples = 32

def get_uncertainty(model, obs, mc_samples=32):
    """计算MC Dropout不确定性"""
    model.set_dropout_mode()
    
    actions = []
    for _ in range(mc_samples):
        action = model.predict(obs, deterministic=True)
        actions.append(action)
    
    model.set_normal_mode()
    
    actions_array = np.array(actions)
    action_std = np.std(actions_array, axis=0)
    uncertainty = np.linalg.norm(action_std)
    
    return uncertainty

print("=" * 60)
print("收集正常回合的MC Dropout不确定性数据")
print("=" * 60)

# 数据收集
uncertainty_data = [] 
safe_episodes_count = 0
total_episodes = 0

i = 0
while safe_episodes_count < 1:  #收集500个安全回合
    obs, _ = env.reset(seed=i)
    terminated, truncated = False, False
    cnt = 0
    total_reward = 0    
    episode_uncertainties = []
    
    while not terminated and not truncated:
        # 跳过前50步
        if cnt < 50:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            total_reward += reward
            continue
            
        uncertainty = get_uncertainty(model, obs, mc_samples)
        episode_uncertainties.append(uncertainty)
        
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        cnt += 1
        total_reward += reward

    total_episodes += 1
    
    # 判断是否为安全回合
    is_safe = not (info.get("out_of_route", False) or info.get('out_of_playfield', False))
    
    if is_safe:
        # 只保存安全回合的不确定性数据
        uncertainty_data.extend(episode_uncertainties)
        safe_episodes_count += 1
        
        # 计算episode统计
        if len(episode_uncertainties) > 0:
            ep_min = np.min(episode_uncertainties)
            ep_max = np.max(episode_uncertainties)
            ep_mean = np.mean(episode_uncertainties)
            ep_samples = len(episode_uncertainties)
        else:
            ep_min = ep_max = ep_mean = 0
            ep_samples = 0
        
        print(f'安全回合 {safe_episodes_count}/500 (总第{total_episodes}回合): '
              f'Steps={cnt}, Reward={total_reward:.1f}, Samples={ep_samples}, '
              f'Unc_range=[{ep_min:.8f}, {ep_max:.8f}], Unc_mean={ep_mean:.8f}')
    else:
        # 不安全回合，跳过
        print(f'不安全回合 (总第{total_episodes}回合): Steps={cnt}, Reward={total_reward:.1f} - 跳过')
    
    i += 1

# 转换为numpy数组
uncertainty_data = np.array(uncertainty_data)

print("\n" + "=" * 60)
print("数据收集完成")
print("=" * 60)

print(f"总尝试回合数: {total_episodes}")
print(f"安全回合数: {safe_episodes_count}")
print(f"不安全回合数: {total_episodes - safe_episodes_count}")
print(f"安全回合比例: {safe_episodes_count/total_episodes:.2%}")

print(f"\n收集的不确定性样本数: {len(uncertainty_data)}")
print(f"不确定性统计:")
print(f"  Min: {np.min(uncertainty_data):.8f}")
print(f"  Max: {np.max(uncertainty_data):.8f}")
print(f"  Mean: {np.mean(uncertainty_data):.8f}")
print(f"  Std: {np.std(uncertainty_data):.8f}")
print(f"  Median: {np.median(uncertainty_data):.8f}")
print(f"  25th percentile: {np.percentile(uncertainty_data, 25):.8f}")
print(f"  75th percentile: {np.percentile(uncertainty_data, 75):.8f}")

# 保存数据
np.save('./MCD/CarRacing_safe_uncertainty_data.npy', uncertainty_data)

# 保存元数据
metadata = {
    'total_episodes': total_episodes,
    'safe_episodes': safe_episodes_count,
    'unsafe_episodes': total_episodes - safe_episodes_count,
    'total_samples': len(uncertainty_data),
    'skip_initial_steps': 50,
    'mc_samples': mc_samples
}
np.save('./MCD/CarRacing_safe_metadata.npy', metadata)

