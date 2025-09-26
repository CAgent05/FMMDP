import argparse
import torch
from collections import deque
from utils import prepare_agent
import numpy as np
import pandas as pd
import os
import warnings
import time
import matplotlib.pyplot as plt

from Model import MTPool


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCSA')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--data_path', type=str, default="/home/jc/CY/FMMDP/data/Train/",
                    help='the path of data.')
parser.add_argument('--gnn', type=str, default="GNN",
                    help='GNN or GIN')
parser.add_argument('--relation', type=str, default="corr",
                    help='dynamic or corr or all_one')
parser.add_argument('--pooling', type=str, default="CoSimPool",
                    help='CoSimPool or MemPool or DiffPool or SAGPool')
parser.add_argument('--use_cuda', type=int, default=1, help='cpu or gpu.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--alg', type=str, default='MTPool', help='Algorithm to use for training.')


# CLI Parse
args = parser.parse_args()

if args.use_cuda==1:
    args.cuda = torch.cuda.is_available()
else:
    args.cuda = False

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.autograd.set_detect_anomaly(True)

# make dir for exp result
result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

# prepare for agent and env
model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pt'
print(model_dir)
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
    
seq_length = args.nsteps


MTPool_model = MTPool(use_cuda=args.cuda,
                      dataset_path=args.data_path,
                      dataset=args.dataset + 'SA_20' ,
                      graph_method=args.gnn,
                      relation_method=args.relation,
                      pooling_method=args.pooling
                      )

# load failure monitoring model
print(model_dir)
MTPool_model.load_state_dict(torch.load(model_dir))
MTPool_model.to('cuda:0')
MTPool_model.eval()

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
            obs_input = torch.cat(list(record), dim=0).transpose(0, 1).unsqueeze(0).to('cuda:0')  # Concatenate along the first dimension
            pre = MTPool_model(obs_input, deploy=True)
            # print(pre)

            label = torch.argmax(pre[0]).item()

            # if pre[0][1].item() >= 0.5:
            #     label = 1
            # else:
            #     label = 0

            
            # if label.item() == 1:
            if label == 1:
                if not is_warning:
                    current_warning_start = cnt
                    is_warning = True
                    prob = pre[0][1].item()
                pre_label[i] = 1

            else:
                if is_warning:
                    last_warning_start = current_warning_start
                    is_warning = False
    
    if is_warning:
        last_warning_start = current_warning_start
    
    if prob == -1 and len(record) == 20:
        obs_input = torch.cat(list(record), dim=0).transpose(0, 1).unsqueeze(0).to('cuda:0')
        pre = MTPool_model(obs_input, deploy=True)
        prob = pre[0][1].item()

        # probs.append(prob)

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
