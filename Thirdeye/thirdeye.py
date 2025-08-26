import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
import torch
import numpy as np
import time
from collections import deque
from utils import get_threshold, Saliency
from vae import VAE
import pickle
import tqdm
import matplotlib.pyplot as plt
import argparse
import warnings 
warnings.filterwarnings('ignore')


def test_model():
    def score_when_decrease(output):
        return -1.0 * output[0][0][0]
    
    env = gym.make("CarRacing-v3",
                render_mode="rgb_array",)
    env = GrayscaleObservation(env, keep_dim=True)
    model = PPO.load("./gymmodel/CarRacing.zip", device='cuda:0')

    ha_max_values = np.load("./Thirdeye/ha_max_values.npy")
    ha_mean_values = np.load("./Thirdeye/ha_mean_values.npy")
    hd_max_values = np.abs(np.load("./Thirdeye/hd_max_values.npy"))
    hd_mean_values = np.abs(np.load("./Thirdeye/hd_mean_values.npy"))
    hrl_max_values = np.load("./Thirdeye/hrl_max_values.npy")
    hrl_mean_values = np.load("./Thirdeye/hrl_mean_values.npy")


    ha_mean_threshold = get_threshold(ha_mean_values, conf_level=0.95)
    hd_mean_threshold = get_threshold(hd_mean_values, conf_level=0.95)
    hrl_mean_threshold = get_threshold(hrl_mean_values, conf_level=0.95)

    ha_max_threshold = get_threshold(ha_max_values, conf_level=0.95)
    hd_max_threshold = get_threshold(hd_max_values, conf_level=0.95)
    hrl_max_threshold = get_threshold(hrl_max_values, conf_level=0.95)


    # 画出不同指标的分布图
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.hist(ha_max_values, bins=50, color='b', alpha=0.7)
    plt.axvline(ha_max_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HA Max Value Distribution")
    plt.xlabel("HA Max Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 4)
    plt.hist(ha_mean_values, bins=50, color='b', alpha=0.7)
    plt.axvline(ha_mean_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HA Mean Value Distribution")
    plt.xlabel("HA Mean Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 2)
    plt.hist(hd_max_values, bins=50, color='b', alpha=0.7)
    plt.axvline(hd_max_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HD Max Value Distribution")
    plt.xlabel("HD Max Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 5)
    plt.hist(hd_mean_values, bins=50, color='b', alpha=0.7)
    plt.axvline(hd_mean_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HD Mean Value Distribution")
    plt.xlabel("HD Mean Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 3)
    plt.hist(hrl_max_values, bins=50, color='b', alpha=0.7)
    plt.axvline(hrl_max_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HRL Max Value Distribution")
    plt.xlabel("HRL Max Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 6)
    plt.hist(hrl_mean_values, bins=50, color='b', alpha=0.7)
    plt.axvline(hrl_mean_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title("HRL Mean Value Distribution")
    plt.xlabel("HRL Mean Value")
    plt.ylabel("Frequency")

    plt.savefig("./Thirdeye/ha_hd_hrl_value.png")
    check_episode = 100

    window_size = 50
        
    labels_mean = {'ha':[], 'hd':[], 'hrl':[]}
    labels_max = {'ha':[], 'hd':[], 'hrl':[]}

    true_label = np.zeros(check_episode)

    for i in tqdm.tqdm(list(range(check_episode))):
        
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        cnt = 0
        saliency_map_record = deque(maxlen=window_size)
        ha_window = []
        hd_window = []
        hrl_window = []
        
        for id in ['ha', 'hd', 'hrl']:
            labels_mean[id].append([])
            labels_max[id].append([])
        
        saliency = Saliency(model.policy)
        vae = VAE()
        vae.load_state_dict(torch.load('/home/cy/PaperWork/FMMDP/Thirdeye/vae.pth'))
        vae.to('cuda:0')

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            cnt += 1

            if cnt >= 50:
                t = time.time()
                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(model.device)
                saliency_map = saliency(score_when_decrease, state, smooth_noise=0.2, smooth_samples=20)
                saliency_map_record.append(saliency_map)

                ha_value = saliency_map.mean().item()  # 计算 HA 值
                ha_window.append(ha_value)  # 将 HA 值添加到滑动窗口
                # print("ha_time", time.time() - t)
                if len(saliency_map_record) > 1:
                    diff = saliency_map_record[-1] - saliency_map_record[-2]
                    hd_value = torch.mean(diff)  # 计算 HD 值
                    hd_window.append(hd_value.item())  # 将 HD 值添加到滑动窗口
                # print("hd_time", time.time() - t)

                # 计算 HRL
                saliency_map_tensor = torch.tensor(saliency_map_record[-1])
                with torch.no_grad():
                    recon_batch, mu, logvar = vae(saliency_map_tensor)
                    hrl_value = torch.nn.functional.mse_loss(saliency_map_tensor.view(-1), recon_batch)
                hrl_window.append(hrl_value.item())


                # 当窗口满时，计算均值和最大值
                if len(ha_window) >= window_size:
                    ha_mean = sum(ha_window[-window_size:]) / window_size  # 计算 HA 窗口均值
                    ha_max = max(ha_window[-window_size:])
                    if ha_mean > ha_mean_threshold:
                        labels_mean['ha'][-1].append(1)
                    else:
                        labels_mean['ha'][-1].append(0)
                    if ha_max > ha_max_threshold:
                        labels_max['ha'][-1].append(1)
                    else:
                        labels_max['ha'][-1].append(0)
                else:
                    labels_mean['ha'][-1].append(0)
                    labels_max['ha'][-1].append(0)
                        

                if len(hd_window) >= window_size:
                    hd_mean = sum(hd_window[-window_size:]) / window_size
                    hd_max = max(hd_window[-window_size:])
                    if hd_mean > hd_mean_threshold:
                        labels_mean['hd'][-1].append(1)
                    else:
                        labels_mean['hd'][-1].append(0)
                    if hd_max > hd_max_threshold:
                        labels_max['hd'][-1].append(1)
                    else:
                        labels_max['hd'][-1].append(0)
                else:
                    labels_mean['hd'][-1].append(0)
                    labels_max['hd'][-1].append(0)

                if len(hrl_window) >= window_size:
                    hrl_mean = sum(hrl_window[-window_size:]) / window_size  # 计算 HRL 窗口均值
                    hrl_max = max(hrl_window[-window_size:])  # 计算 HRL 窗口最大值
                    if hrl_mean > hrl_mean_threshold:
                        labels_mean['hrl'][-1].append(1)
                    else:
                        labels_mean['hrl'][-1].append(0)
                    if hrl_max > hrl_max_threshold:
                        labels_max['hrl'][-1].append(1)
                    else:
                        labels_max['hrl'][-1].append(0)
                else:
                    labels_mean['hrl'][-1].append(0)
                    labels_max['hrl'][-1].append(0)
        
        

        # 计算预测提前量

        # features.append(todeynet.get_features().detach().cpu().numpy())
        if info.get("out_of_route", False):
            true_label[i] = 1
        elif info.get('out_of_playfield', False):
            true_label[i] = 1
        else:
            true_label[i] = 0

        # print(labels_mean['ha'][-1],labels_mean['hd'][-1],labels_mean['hrl'][-1], labels_max['ha'][-1],labels_max['hd'][-1],labels_max['hrl'][-1])
        
    with open('./Thirdeye/labels_20.pkl', 'wb') as f:
        pickle.dump((true_label, labels_mean, labels_max), f)
        

def analyze_steps(data):
    i = len(data) - 1
    steps = len(data) - 1
    failure_predict = False
    while i:
        if not failure_predict and data[i]:
            failure_predict = True
            steps = len(data) - 1 - i
        if failure_predict:
            if data[i]:
                steps = len(data) - 1 - i
            else:
                break
        i = i - 1
    return steps


def read_results():
    with open('./Thirdeye/labels_20.pkl', 'rb') as f:
        true_label, labels_mean, labels_max = pickle.load(f)
    index = [
        {'TP': 0, 'FN': 1, 'FP': 2, 'TN': 3},
        {'ha': 0, 'hd': 1, 'hrl': 2},
        {'mean': 0, 'max': 1}
    ]
    result = np.zeros([len(idx) for idx in index])
    TP_steps = {}
    for method in ['mean', 'max']:
        TP_steps[method] = {}
        for id in ['ha', 'hd', 'hrl']:
            TP_steps[method][id] = []

    labels = {'mean': labels_mean, 'max': labels_max}
    for k in range(len(true_label)):
        for method in ['mean', 'max']:
            for id in ['ha', 'hd', 'hrl']:
                data = np.array(labels[method][id][k])
                if true_label[k]:
                    if data.any():
                        result[index[0]['TP'], index[1][id], index[2][method]] += 1
                        TP_steps[method][id].append(analyze_steps(data))
                    else:
                        result[index[0]['FN'], index[1][id], index[2][method]] += 1
                else:
                    if data.any():
                        result[index[0]['FP'], index[1][id], index[2][method]] += 1
                    else:
                        result[index[0]['TN'], index[1][id], index[2][method]] += 1
    
    print('\t\tTP\tFP\tFN\tTN\tpr\trecall\tfpr\tf1\tTP_steps')
    for method in ['mean', 'max']:
        for id in ['ha', 'hd', 'hrl']:
            item = result[:, index[1][id], index[2][method]]
            precision = item[index[0]['TP']] / (item[index[0]['TP']] + item[index[0]['FP']])
            recall = item[index[0]['TP']] / (item[index[0]['TP']] + item[index[0]['FN']])
            f1 = 2*precision*recall / (precision + recall)
            fpr = item[index[0]['FP']] / (item[index[0]['FP']] + item[index[0]['TN']])
            print(f"{id}_{method}  \t{item[index[0]['TP']]/5}\t{item[index[0]['FP']]/5}\t{item[index[0]['FN']]/5}\t{item[index[0]['TN']]/5}\t{precision:.3}\t{recall:.3}\t{fpr:.3}\t{f1:.3}\t{np.mean(TP_steps[method][id])}")


def plot_item(number = 0):
    with open('./Thirdeye/labels_20.pkl', 'rb') as f:
        true_label, labels_mean, labels_max = pickle.load(f)
    k = 1
    labels = {'mean': labels_mean, 'max': labels_max}
    fig = plt.figure(figsize=(11.5,7))
    ax = fig.add_axes([0.02, 0.07, 0.95, 0.75], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('step', fontsize=15)
    for method in ['mean', 'max']:
        for id in ['ha', 'hd', 'hrl']:
            plt.subplot(2, 3, k)
            plt.plot(labels[method][id][number], color='red' if true_label[number] else 'green')
            k = k + 1
            plt.title(f'{id}_{method}')
    plt.subplot(2,3,1)
    plt.ylabel('predicted failure label', fontsize=12)
    plt.subplot(2,3,4)
    plt.ylabel('predicted failure label', fontsize=12)
    plt.savefig(f'{number}_10.jpg')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('-s', '--state', metavar='DATASET', default='test')
    args = parser.parse_args()
    
    if args.state == 'test':
        test_model()
    elif args.state == 'analyze':
        read_results()
        for k in range(3):
            plot_item(k)