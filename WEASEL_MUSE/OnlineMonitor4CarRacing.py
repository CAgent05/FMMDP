import argparse
import torch
from collections import deque
from util import prepare_agent
import numpy as np
import pandas as pd
import pickle
import time
import os


parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCAC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="WEASEL", help='the algorithm used for training')
args = parser.parse_args()

model_dir = './model/WEASEL/' + args.dataset + '_' + str(args.nsteps) + '/'

result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

if args.dataset[-3:] == "SAR":
    input_tag = "SAR"
    args.dataset = args.dataset[:-3]
elif args.dataset[-2:] == 'SA':
    input_tag = 'SA'
    args.dataset = args.dataset[:-2]
else:
    input_tag = "S"
    args.dataset = args.dataset[:-1]

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

env, model, num_nodes, alg_tag = prepare_agent(args.dataset, input_tag)

print(f"The dim of features: {num_nodes}\nThe alg used for training agent: {alg_tag}")

df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps', 'T'])
    
seq_length = args.nsteps


with open(model_dir + 'weaselmuse.pkl', 'rb') as f:
    weaselmuse = pickle.load(f)
    
with open(model_dir + 'RandomForest.pkl', 'rb') as f:
    RandomForest = pickle.load(f)

check_episode = args.episodes

pre_label = np.zeros(check_episode)
true_label = np.zeros(check_episode)

# # for i in range(100):
for i in range(check_episode):
    
    seed = np.random.randint(5000, 10000)
    obs, _ = env.reset(seed=seed)
    
    done = False
    truncated = False
    total_reward = 0
    record = deque(maxlen=seq_length)
    cnt = 0
    prob = -1
    steps = 0

    is_warning = False
    last_warning_start = 0
    current_warning_start = 0
    
    t = time.time()
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if cnt >= 50:
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(model.device)
            feature = model.policy.extract_features(state)
            act = np.array(action).reshape(1, -1)
            feature = feature.detach().cpu().numpy()
            record.append(np.concatenate((feature, act), axis=1))

        cnt += 1
        if len(record) == seq_length:
            obs_input = np.array(record)
            obs_input = obs_input.transpose(1, 2, 0)
            if args.dataset == 'CarRacing':
                nonzero_feature_indices = [0, 8, 9, 10, 13, 14, 16, 17, 19, 20, 21, 22, 24, 27, 31, 32, 33, 34]
                obs_input = obs_input[:, nonzero_feature_indices, :]
            obs_input = weaselmuse.transform(obs_input)
            a = RandomForest.predict_proba(obs_input)
            label = np.argmax(a)

            if label.item() == 1:
                if not is_warning:
                    current_warning_start = cnt
                    is_warning = True
                pre_label[i] = 1
                prob =  np.exp(a) / np.sum(np.exp(a))
                prob = prob[0][1]
                steps = cnt

            else:
                if is_warning:
                    last_warning_start = current_warning_start
                    is_warning = False
    
    if is_warning:
        last_warning_start = current_warning_start
    
    if prob == -1 and len(record) == 20:
        a = RandomForest.predict_proba(obs_input)
        pre_label[i] = 1
        prob =  np.exp(a) / np.sum(np.exp(a))
        prob = prob[0][1]
        steps = cnt
                    

    if args.dataset == 'BipedalWalkerHC':
        if total_reward < 285:
            true_label[i] = 1
            if pre_label[i] == 1 and last_warning_start > 0:
                steps = cnt - last_warning_start
    else:   
        if cnt < 1000:
            true_label[i] = 1
            if pre_label[i] == 1 and last_warning_start > 0:
                steps = cnt - last_warning_start
                
    if steps == 0:
        steps = cnt
    t = (time.time() - t) / cnt
    print(f"Episode: {i}\tReward: {total_reward:5.2f}\tPre: {pre_label[i]:2}\tTrue: {true_label[i]:2}\tProb: {prob:.4f}\tSteps: {steps:4d}\tT: {t:.4f}")

    df.loc[len(df)] = [i, total_reward, pre_label[i], true_label[i], prob, steps, t]

df.to_csv(result_save_dir, index=False)