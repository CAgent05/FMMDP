import numpy as np
from tools import make_env_model, collect
import torch
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Data Collection')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=3000)

args = parser.parse_args()

n = args.nsteps

episodes = args.episodes

# Box2d
env, model = make_env_model(args.dataset)

# label= 1 indicate failure
timeseries = []
labels = []
a = time.time()
for i in range(episodes):
    t = time.time()
    done = False
    truncated = False
    total_reward = 0
    obs, _ = env.reset(seed=i)
    record = []
    cnt = 0
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        state = torch.as_tensor(obs)
        actions = torch.as_tensor(action)
        obs, reward, done, truncated, info = env.step(action)
        rew = torch.as_tensor(reward)
        state, actions, rew = state.view(1, -1), actions.view(1, -1), rew.view(1, -1)
        # print('state:', state.shape, 'action:', actions.shape, 'reward:', rew.shape)
        record.append(torch.cat([state, actions, rew], dim=1))
        total_reward += reward
        cnt += 1
    # print(i+1, cnt, total_reward, reward, time.time()-t)
    collect(args.dataset, n, total_reward, record, timeseries, labels, cnt=cnt, info=info)
    
    print(f'Epoch:{i+1}\tSteps:{cnt}\tReward:{total_reward:.2f}\tLabel:{labels[-1]}\tT:{time.time()-t:.2f}')

print('Total time:', time.time() - a)

timeseries = torch.stack(timeseries, dim=0)
labels = torch.tensor(labels, dtype=torch.long)
print('TS_shape:', list(timeseries.shape), 'Fail_num:', labels.sum().item())

X_train, X_valid = timeseries[:2000], timeseries[2000:3000]
y_train, y_valid = labels[:2000], labels[2000:3000]

X_train = X_train.permute(0, 1, 3, 2)
X_valid = X_valid.permute(0, 1, 3, 2)

if args.dataset =='Humanoid':
    X_train = torch.cat((X_train[:, :, :45, :], X_train[:, :, 376:, :]), dim=2)
    X_valid = torch.cat((X_valid[:, :, :45, :], X_valid[:, :, 376:, :]), dim=2)

print('X_train_SAR shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
print('X_valid_SAR_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

save_dir = './data/' + 'Train/' + args.dataset + 'SAR' + '_' + str(n) + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save state, action, reward data
torch.save(X_train, save_dir + 'X_train.pt')
torch.save(y_train, save_dir + 'y_train.pt')
torch.save(X_valid, save_dir + 'X_valid.pt')
torch.save(y_valid, save_dir + 'y_valid.pt')


# save state, action
if n == 20:
    save_dir = './data/' + 'Train/' + args.dataset + 'SA' + '_' + str(n) + '/'
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
    save_dir = './data/' + 'Train/' + args.dataset+ 'S' + '_' + str(n) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.dataset == 'BipedalWalkerHC':
        X_train = X_train[:, :, :24, :]
        X_valid = X_valid[:, :, :24, :]
    elif args.dataset == 'InvertedDoublePendulum':
        X_train = X_train[:, :, :11, :]
        X_valid = X_valid[:, :, :11, :]
    elif args.dataset == 'Walker2d':
        X_train = X_train[:, :, :17, :]
        X_valid = X_valid[:, :, :17, :]
    elif args.dataset == 'Hopper':
        X_train = X_train[:, :, :11, :]
        X_valid = X_valid[:, :, :11, :]
    elif args.dataset == 'Humanoid':
        X_train = X_train[:, :, :45, :]
        X_valid = X_valid[:, :, :45, :]

    
    print('X_train_S_shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
    print('X_valid_S_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

    torch.save(X_train, save_dir + 'X_train.pt')
    torch.save(X_valid, save_dir + 'X_valid.pt')
    torch.save(y_train, save_dir + 'y_train.pt')
    torch.save(y_valid, save_dir + 'y_valid.pt')
