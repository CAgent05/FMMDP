import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
import torch
import numpy as np
import time
from collections import deque
import pandas as pd
from PIL import Image
from utils import get_threshold, Saliency
import matplotlib.pyplot as plt
from vae import VAE
from torch import nn

def score_when_decrease(output):
    return -1.0 * output[0][0][0]

df = pd.DataFrame(columns=['epoch', 'steps', 'reward', 'label', 'time', 'info'])

env = gym.make("CarRacing-v3",
               render_mode="rgb_array",)
env = GrayscaleObservation(env, keep_dim=True)

model = PPO.load("./gymmodel/CarRacing.zip", device='cuda:0')
print(model.device)

labels = []

# 初始化滑动窗口
window_size = 20


# 用于保存窗口的均值和最大值
ha_mean_values = []
ha_max_values = []
hd_mean_values = []
hd_max_values = []
hrl_mean_values = []
hrl_max_values = []

for i in range(100):
    obs, _ = env.reset(seed=i)
    done, truncated = False, False
    cnt = 0
    total_reward = 0    
    record = []
    t = time.time()
    saliency_map_record = deque(maxlen=window_size)
    ha_window = deque(maxlen=window_size)  # 用于存储 HA 值的滑动窗口
    hd_window = deque(maxlen=window_size)  # 用于存储 HD 值的滑动窗口
    hrl_window = deque(maxlen=window_size)  # 用于存储 HRL 值的滑动窗口
    
    saliency = Saliency(model.policy)
    vae = VAE()
    vae.load_state_dict(torch.load('./Thirdeye/vae.pth', weights_only=True, map_location='cuda:0'))
    
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        # print(obs, reward, done, truncated, info)
        
        cnt += 1
        if cnt >= 50:
            t = time.time()
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(model.device)
            saliency_map = saliency(score_when_decrease, state, smooth_noise=0.2, smooth_samples=20)
            
            saliency_map = saliency_map.squeeze().cpu().numpy()
      
            saliency_map_record.append(saliency_map)
            
            ha_value = saliency_map.mean().item()  # 计算 HA 值
            ha_window.append(ha_value)  # 将 HA 值添加到滑动窗口

            # 计算 HD
            if len(saliency_map_record) > 1:
                diff = saliency_map_record[-1] - saliency_map_record[-2]
                hd_value = np.mean(diff)  # 计算 HD 值
                hd_window.append(hd_value)  # 将 HD 值添加到滑动窗口

            # 计算 HRL
            saliency_map_tensor = torch.tensor(saliency_map_record[-1])
            with torch.no_grad():
                recon_batch, mu, logvar = vae(saliency_map_tensor)
                hrl_value = nn.functional.mse_loss(saliency_map_tensor.view(1,-1), recon_batch)
            hrl_window.append(hrl_value.item())

            # 当窗口满时，计算均值和最大值
            if len(ha_window) == window_size:
                ha_mean = sum(ha_window) / window_size  # 计算 HA 窗口均值
                ha_max = max(ha_window)  # 计算 HA 窗口最大值
                ha_mean_values.append(ha_mean)
                ha_max_values.append(ha_max)

            if len(hd_window) == window_size:
                hd_mean = sum(hd_window) / window_size  # 计算 HD 窗口均值
                hd_max = max(hd_window)  # 计算 HD 窗口最大值
                hd_mean_values.append(hd_mean)
                hd_max_values.append(hd_max)

            if len(hrl_window) == window_size:
                hrl_mean = sum(hrl_window) / window_size  # 计算 HRL 窗口均值
                hrl_max = max(hrl_window)  # 计算 HRL 窗口最大值
                hrl_mean_values.append(hrl_mean)
                hrl_max_values.append(hrl_max)
            
        total_reward += reward



    if info.get("out_of_route", False):
        labels.append(1)
    elif info.get('out_of_playfield', False):
        labels.append(1)
    else:
        labels.append(0)  
        
        
    print('epoch:', i+1, 'steps:', cnt, 'reward:', total_reward, 'label:', labels[-1], 't:', time.time()-t, "info:", info)

ha_mean_values = np.array(ha_mean_values)
ha_max_values = np.array(ha_max_values)
hd_mean_values = np.array(hd_mean_values)
hd_max_values = np.array(hd_max_values)
hrl_mean_values = np.array(hrl_mean_values)
hrl_max_values = np.array(hrl_max_values)

print(ha_mean_values.shape, ha_max_values.shape, hd_mean_values.shape, hd_max_values.shape, hrl_mean_values.shape, hrl_max_values.shape)

# 保存 HA 和 HD 的均值和最大值
np.save("./Thirdeye/ha_mean_values.npy", ha_mean_values)
np.save("./Thirdeye/ha_max_values.npy", ha_max_values)
np.save("./Thirdeye/hd_mean_values.npy", hd_mean_values)
np.save("./Thirdeye/hd_max_values.npy", hd_max_values)
np.save("./Thirdeye/hrl_mean_values.npy", hrl_mean_values)
np.save("./Thirdeye/hrl_max_values.npy", hrl_max_values)