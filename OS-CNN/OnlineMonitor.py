import argparse
import torch
from collections import deque
from util import OS_CNN, prepare_agent, transform_input, generate_layer_parameter_list
import numpy as np
import pandas as pd
import time
import os
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCAC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="OS-CNN", help='the algorithm used for training')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
args = parser.parse_args()

model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pth'

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

env, model, num_nodes, alg_tag = prepare_agent(args.dataset, input_tag=input_tag)
print(f"The dim of features: {num_nodes}\nThe alg used for training agent: {alg_tag}")

args.kern_size = [ int(l) for l in args.kern_size.split(",") ]

df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps', 'T'])
    
seq_length = 20
Max_kernel_size = 89


quarter_or_half = 4
receptive_field_shape= min(int(seq_length/quarter_or_half), Max_kernel_size)
start_kernel_size = 1
paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128]
n_class = 2

n_input_channel = num_nodes


layer_parameter_list = generate_layer_parameter_list(start_kernel_size,receptive_field_shape, paramenter_number_of_layer_list,in_channel = 1)
torch_OS_CNN = OS_CNN(layer_parameter_list, n_class ,n_input_channel, True).to(device)

torch_OS_CNN.load_state_dict(torch.load(model_dir))  

torch_OS_CNN.to('cuda:0')
torch_OS_CNN.eval()

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

    last_warning_start = 0
    current_warning_start = 0
    is_warning = False
    
    t = time.time()
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        record = transform_input(obs, action, model, record, alg_tag)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        cnt += 1
        if len(record) == seq_length:
            obs_input = torch.stack(list(record), dim=1)
            obs_input = obs_input.permute(0, 2, 1).float().to('cuda:0')
            a = torch_OS_CNN(obs_input)
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
        a = torch_OS_CNN(obs_input)
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
    

