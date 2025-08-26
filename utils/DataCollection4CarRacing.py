import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
import torch
import numpy as np
import time
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='Data Collection')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=3000)
args = parser.parse_args()

df = pd.DataFrame(columns=['epoch', 'steps', 'reward', 'label', 'time', 'info'])

env = gym.make("CarRacing-v3",
               render_mode="rgb_array",)
env = GrayscaleObservation(env, keep_dim=True)

model = PPO.load("./gymmodel/CarRacing.zip")

timeseries = []
labels = []

n = args.nsteps
episodes = args.episodes

for i in range(episodes):
    obs, _ = env.reset(seed=i)
    done, truncated = False, False
    cnt = 0
    total_reward = 0    
    record = []
    t = time.time()
    
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        cnt += 1
        if cnt >= 50:
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(model.device)   
            feature = model.policy.extract_features(state).to('cpu')
            rew = torch.as_tensor(reward).view(1, -1)
            act = torch.as_tensor(action).view(1, -1)
            record.append(torch.cat([feature, act, rew], dim=1))

        total_reward += reward


    if info.get("out_of_route", False):
        timeseries.append(torch.stack(record[-n:], dim=1))
        labels.append(1)
    elif info.get('out_of_playfield', False):
        timeseries.append(torch.stack(record[-n:], dim=1))
        labels.append(1)
    elif info.get("lap_finished", False):
        index = np.random.randint(0, len(record) - n)
        timeseries.append(torch.stack(record[index:index + n], dim=1))
        labels.append(0)
    else:
        info['out_of_time'] = True
        index = np.random.randint(0, len(record) - n)
        timeseries.append(torch.stack(record[index:index + n], dim=1))
        labels.append(0)  
        
        
    print(f'Epoch:{i+1}\tSteps:{cnt}\tReward:{total_reward:.2f}\tLabel:{labels[-1]}\tT:{time.time()-t:.2f}\tInfo:{info}')
    df.loc[len(df)] = [i+1, cnt, total_reward, labels[-1], time.time()-t, info]

timeseries = torch.stack(timeseries, dim=0).permute(0, 1, 3, 2)
labels = torch.tensor(labels, dtype=torch.long).to('cpu')
print('TS_shape:', list(timeseries.shape), 'Fail_num:', labels.sum().item())

X_train, X_valid = timeseries[:2000], timeseries[2000:3000]
y_train, y_valid = labels[:2000], labels[2000:3000]


print('X_train_SAR shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
print('X_valid_SAR_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())


save_dir = './data/' + 'Train/CarRacingSAR' + '_' + str(n) + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


torch.save(X_train, save_dir + 'X_train.pt')
torch.save(X_valid, save_dir + 'X_valid.pt')
torch.save(y_train, save_dir + 'y_train.pt')
torch.save(y_valid, save_dir + 'y_valid.pt')


# save state, action
if n == 20:
    save_dir = './data/' + 'Train/CarRacingSA' + '_' + str(n) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X_train = X_train[:, :, :-1, :]
    X_valid = X_valid[:, :, :-1, :]

    print('X_train_SA_shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
    print('X_valid_SA_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

    torch.save(X_train, save_dir + 'X_train.pt')
    torch.save(X_valid, save_dir + 'X_valid.pt')
    torch.save(y_train, save_dir + 'y_train.pt')
    torch.save(y_valid, save_dir + 'y_valid.pt')

# save state    
if n == 20:
    save_dir = './data/' + 'Train/CarRacingS' + '_' + str(n) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    X_train = X_train[:, :, :32, :]
    X_valid = X_valid[:, :, :32, :]

    
    print('X_train_S_shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
    print('X_valid_S_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

    torch.save(X_train, save_dir + 'X_train.pt')
    torch.save(X_valid, save_dir + 'X_valid.pt')
    torch.save(y_train, save_dir + 'y_train.pt')
    torch.save(y_valid, save_dir + 'y_valid.pt')
