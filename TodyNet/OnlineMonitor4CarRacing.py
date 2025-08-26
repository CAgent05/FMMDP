import argparse
import torch
from collections import deque
from utils import prepare_agent
import numpy as np
from src.net import GNNStack
from src.utils import AverageMeter
import pandas as pd
import os
import warnings
from PIL import Image
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='CarRacingSA')
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

args = parser.parse_args()

#prepare for todynet model
model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pth'

result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

args.kern_size = [ int(l) for l in args.kern_size.split(",") ]

df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps', 'T'])

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
print(f"The dim of features: {num_nodes},\nThe alg used for training agent: {alg_tag}")
    
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

todeynet.load_state_dict(torch.load(model_dir))  
todeynet.to('cuda:0')
todeynet.eval()

check_episode = args.episodes

pre_label = np.zeros(check_episode)
true_label = np.zeros(check_episode)

features = []
save_cnt = 0
pic = []



# # for i in range(100):
for i in range(check_episode):
    t = time.time()
    seed = np.random.randint(5000, 10000)
    
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0
    record = deque(maxlen=20) #seq_length
    cnt = 0
    prob = -1
    steps = 0

    last_warning_start = 0
    current_warning_start = 0
    is_warning = False
    
    timerec = []
    probs = []
    
    t = time.time()
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        pic.append(env.render())
        

        # the observation of  first 50 steps are small
        if cnt >= 50:
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(model.device)
            feature = model.policy.extract_features(state)
            act = torch.as_tensor(action).view(1, -1).to("cuda:0") 
            reward = torch.as_tensor(reward).view(1, -1).to("cuda:0")
            if input_tag == 'SAR':
                record.append(torch.cat([feature, act, reward], dim=1))
            elif input_tag == 'SA':
                record.append(torch.cat([feature, act], dim=1))
            else:
                record.append(torch.cat([feature], dim=1))

        cnt += 1
        
        if len(record) == 20:
            obs_input = torch.cat(list(record), dim=0).unsqueeze(0)  # Concatenate along the first dimension
            obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to('cuda:0')
            a = todeynet(obs_input)
            label = torch.argmax(a, dim=1)

            probs.append(torch.softmax(a, dim=1)[0][1].item())
            if label.item() == 1:
                # 如果之前不是预警状态，记录当前预警开始点
                if not is_warning:
                    current_warning_start = cnt
                    is_warning = True
                # 更新预测标签和概率（每次都更新）
                pre_label[i] = 1
                prob = torch.softmax(a, dim=1)[0][1].item()
                save_cnt += 1
            else:
                # 如果预警结束，更新最后一次预警的开始点
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

    # features.append(todeynet.get_features().detach().cpu().numpy())
    if info.get("out_of_route", False):
        true_label[i] = 1
        if pre_label[i] == 1 and last_warning_start > 0:
            steps = cnt - last_warning_start
    elif info.get('out_of_playfield', False):
        true_label[i] = 1
        if pre_label[i] == 1 and last_warning_start > 0:
            steps = cnt - last_warning_start
    else:
        true_label[i] = 0
        
    if steps == 0:
        steps = cnt

    t = (time.time()-t) / cnt

    print(f"Episode: {i+1}\tReward: {total_reward:5.2f}\tPre: {pre_label[i]:2}\tTrue: {true_label[i]:2}\tProb: {prob:.2f}\tSteps: {steps:4d}\tT: {t:.4f}")
    
    df.loc[len(df)] = [i, total_reward, pre_label[i], true_label[i], prob, steps, t]

df.to_csv(result_save_dir, index=False)

probs = np.array(probs)

if args.episodes == 1:
    plt.figure(figsize=(10, 5))
    x = np.arange(0, cnt - 69)
    
    # 绘制基础曲线（蓝色）
    plt.plot(x, probs, color='blue', label='Probability (before alert)', alpha=0.7)
    
    # 画 0.5 的水平参考线
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    
    # 找到第一个 prob > 0.5 的点和对应 Step
    first_cross_mask = probs > 0.5
    if np.any(first_cross_mask):  # 确保存在 prob > 0.5 的点
        first_cross_idx = np.argmax(first_cross_mask)
        first_cross_step = x[first_cross_idx]
        first_cross_prob = probs[first_cross_idx]
        
        # 标记预警点（红色五角星）
        plt.scatter(
            last_warning_start-70, first_cross_prob,
            color='red', marker='*', s=200,
            zorder=10,
            label=f'Alert Point (Step {last_warning_start-70})'
        )
        
        # 绘制预警点之后的曲线（橙色）
        plt.plot(x[last_warning_start-70:], probs[last_warning_start-70:], 
                color='orange', label='Probability (after alert)')
        
        # 标记最后一个点（绿色圆圈）
        last_step = x[-1]
        last_prob = probs[-1]
        plt.scatter(
            last_step, last_prob,
            color='green', marker='o', s=100,
            zorder=10,
            label=f'Last Point (Step {last_step}, Prob {last_prob:.2f})'
        )
        
        # 添加预警点文字标注
        plt.annotate(
            f'Alert: Step {last_warning_start-70}\nProb: {first_cross_prob:.2f}',
            xy=(last_warning_start-70, first_cross_prob),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black')
        )
        
        # 添加最后点文字标注
        plt.annotate(
            f'End: Step {last_step}\nProb: {last_prob:.2f}',
            xy=(last_step, last_prob),
            xytext=(-60, -30),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    # 图表装饰
    plt.xlabel('Steps')
    plt.ylabel('Probabilities')
    plt.title('Probability Trend with Alert Point')
    plt.grid(linestyle=':', alpha=0.7)
    plt.legend(loc='best')  # 自动选择最佳图例位置
    plt.tight_layout()
    plt.savefig('./result/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.png')
    plt.show()

img = Image.fromarray(np.array(pic[last_warning_start]))
img.save('./result/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + 'alert.png')

img = Image.fromarray(np.array(pic[-1]))
img.save('./result/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + 'end.png')