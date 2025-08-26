import argparse
import torch
from collections import deque
from utils import prepare_agent
import numpy as np
from src.net import GNNStack
import pandas as pd
import os
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCSA')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')

# CLI Parse
args = parser.parse_args()

# make dir for exp result
result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

# prepare for agent and env
model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pth'
if args.dataset[-3:] == "SAR":
    input_tag = "SAR"
    args.dataset = args.dataset[:-3]
elif args.dataset[-2:] == 'SA':
    input_tag = 'SA'
    args.dataset = args.dataset[:-2]
else:
    input_tag = "S"
    args.dataset = args.dataset[:-1]

env, model, num_nodes, alg_tag = prepare_agent(args.dataset, input_tag)
print(f"The dim of features: {num_nodes}\nThe alg used for training agent: {alg_tag}")

df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps', 'T'])

args.kern_size = [ int(l) for l in args.kern_size.split(",") ]
    
seq_length = args.nsteps

# Model initialisation
todeynet = GNNStack(gnn_model_type=args.arch, 
                    num_layers=args.num_layers, 
                    groups=args.groups, 
                    pool_ratio=args.pool_ratio, 
                    kern_size=args.kern_size, 
                    in_dim=args.in_dim, 
                    hidden_dim=args.hidden_dim, 
                    out_dim=args.out_dim, 
                    seq_len=seq_length, 
                    num_nodes=num_nodes, 
                    num_classes=2)

# load failure monitoring model

todeynet.load_state_dict(torch.load(model_dir))  #r'/home/cy/WorkForISSRE/code/model/InvertedDoublePendulumACVA.pth'
todeynet.to('cuda:0')
todeynet.eval()


check_episode = args.episodes

# Experiment initialisation
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
    probs = []
    
    t = time.time()
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        state = torch.as_tensor(obs)
        actions = torch.as_tensor(action)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        rewards = torch.as_tensor([reward], dtype=torch.float32)  # Ensure rewards is float32

        if input_tag == 'SAR':
            record.append(torch.cat([state.view(1, -1)[:, :45].float(), actions.view(1, -1).float(), rewards.view(1, -1).float()], dim=1))
        elif input_tag == 'SA':
            record.append(torch.cat([state.view(1, -1)[:, :45].float(), actions.view(1, -1).float()], dim=1))
        else:
            record.append(torch.cat([state.view(1, -1)[:, :45].float()], dim=1))

        cnt += 1
        
        if len(record) == args.nsteps:
            obs_input = torch.cat(list(record), dim=0).unsqueeze(0)  # Concatenate along the first dimension
            obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to('cuda:0')
            a = todeynet(obs_input)
            # print()
            probs.append(torch.softmax(a, dim=1)[0][1].item())
            label = torch.argmax(a, dim=1)
            
            if label.item() == 1:
                if not is_warning:
                    current_warning_start = cnt
                    is_warning = True
                pre_label[i] = 1
                prob = torch.softmax(a, dim=1)[0][1].item()

            else:
                if is_warning:
                    last_warning_start = current_warning_start
                    is_warning = False
    
    if is_warning:
        last_warning_start = current_warning_start
    
    if prob == -1 and len(record) == 20:
        obs_input = torch.cat(list(record), dim=0).unsqueeze(0)
        obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to('cuda:0')
        a = todeynet(obs_input)
        prob = torch.softmax(a, dim=1)[0][1].item()
                    

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

probs = np.array(probs)
