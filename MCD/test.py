import torch
import torch.nn as nn
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation
from scipy import stats
import time
import pandas as pd
import os
import warnings
from collections import deque

from utils import PPODropoutInjector

warnings.filterwarnings("ignore")

# make dir for exp result
result_save_dir = './result/MCD/'
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
result_save_dir = result_save_dir + 'CarRacing_MCD_99_windowed.csv'

# 加载已保存的不确定性数据
print("Loading existing uncertainty data...")
uncertainty_data = np.load('./MCD/CarRacing_safe_uncertainty_data.npy')
print(f"Loaded {len(uncertainty_data)} uncertainty samples")

print("\n" + "=" * 60)
print("Calculating MCD Threshold using Gamma Distribution")
print("=" * 60)

# 确保数据为正数（伽马分布要求）
if np.any(uncertainty_data <= 0):
    print("Warning: Found non-positive uncertainty values, adding small positive offset")
    uncertainty_data = uncertainty_data + 1e-8

# 拟合伽马分布
shape, loc, scale = stats.gamma.fit(uncertainty_data, floc=0)

print(f"Gamma Distribution Parameters:")
print(f"  Shape (alpha): {shape:.6f}")
print(f"  Location: {loc:.6f}")
print(f"  Scale (beta): {scale:.6f}")
print(f"  Mean: {shape * scale:.6f}")
print(f"  Std: {np.sqrt(shape * scale**2):.6f}")

# 拟合优度检验
ks_statistic, ks_p_value = stats.kstest(uncertainty_data, 
                                        lambda x: stats.gamma.cdf(x, shape, loc, scale))
print(f"\nKolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_statistic:.6f}")
print(f"  P-value: {ks_p_value:.6f}")
print(f"  Fit Quality: {'Good' if ks_p_value > 0.05 else 'Needs Improvement'}")

# 计算不同置信水平的阈值
confidence_levels = [90, 95, 97, 99, 99.5, 99.9]
thresholds = {}

print(f"\nGamma Distribution Thresholds:")
for conf in confidence_levels:
    threshold = stats.gamma.ppf(conf/100, shape, loc, scale)
    thresholds[conf] = threshold
    print(f"  {conf}% confidence: {threshold:.8f}")

# 选择99%置信水平作为检测阈值
selected_threshold = thresholds[99]
print(f"\nSelected Threshold (99% confidence): {selected_threshold:.8f}")

# 验证阈值在训练数据上的表现
detection_rate = np.mean(uncertainty_data > selected_threshold)
theoretical_rate = 1 - stats.gamma.cdf(selected_threshold, shape, loc, scale)
print(f"Training Data Detection Rate: Actual={detection_rate:.4f}, Theoretical={theoretical_rate:.4f}")

# 设置环境进行在线验证
env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = GrayscaleObservation(env, keep_dim=True)

model = PPODropoutInjector(
    "./gymmodel/CarRacing.zip",
    cnn_dropout_rate=0.05,
    mlp_dropout_rate=0.05,
    env=env
)

mc_samples = 32
window_size = 20  # 滑动窗口大小

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

print("\n" + "=" * 60)
print("Online Deployment with Failure Prediction (20-step windowed)")
print("=" * 60)

# 创建DataFrame保存结果
df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre_Mean', 'Pre_Max', 'True', 'Uncertainty_Mean', 'Uncertainty_Max', 'Steps', 'T'])

# 在线验证
deployment_episodes = 1000
all_uncertainties = []
episode_results = []
pre_label_mean = np.zeros(deployment_episodes)
pre_label_max = np.zeros(deployment_episodes)
true_label = np.zeros(deployment_episodes)

for i in range(deployment_episodes):
    obs, _ = env.reset(seed=i + 5000)  # 使用不同的seed确保是新数据
    terminated, truncated = False, False
    cnt = 0
    total_reward = 0
    episode_uncertainties = []
    
    # 失效预测相关变量
    is_warning_mean = False
    is_warning_max = False
    last_warning_start_mean = 0
    last_warning_start_max = 0
    current_warning_start_mean = 0
    current_warning_start_max = 0
    steps = 0
    uncertainty_mean = -1  # 记录最后一次不确定性均值
    uncertainty_max = -1   # 记录最后一次不确定性最大值
    
    # 滑动窗口
    uncertainty_window = deque(maxlen=window_size)
    
    start_time = time.time()
    
    while not terminated and not truncated:
        if cnt < 50:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            total_reward += reward
            continue
        
        # 计算当前步的不确定性
        current_uncertainty = get_uncertainty(model, obs, mc_samples)
        uncertainty_window.append(current_uncertainty)
        episode_uncertainties.append(current_uncertainty)
        all_uncertainties.append(current_uncertainty)
        
        # 如果窗口还没填满，使用已有数据
        if len(uncertainty_window) == 20:
            window_uncertainties = list(uncertainty_window)
            uncertainty_mean = np.mean(window_uncertainties)
            uncertainty_max = np.max(window_uncertainties)
            
            # 失效预测：基于均值
            if uncertainty_mean > selected_threshold:
                if not is_warning_mean:
                    current_warning_start_mean = cnt
                    is_warning_mean = True
                pre_label_mean[i] = 1
            else:
                if is_warning_mean:
                    last_warning_start_mean = current_warning_start_mean
                    is_warning_mean = False
            
            # 失效预测：基于最大值
            if uncertainty_max > selected_threshold:
                if not is_warning_max:
                    current_warning_start_max = cnt
                    is_warning_max = True
                pre_label_max[i] = 1
            else:
                if is_warning_max:
                    last_warning_start_max = current_warning_start_max
                    is_warning_max = False
        
        # 执行动作
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        cnt += 1
        total_reward += reward

    # 如果结束时仍在预警状态
    if is_warning_mean:
        last_warning_start_mean = current_warning_start_mean
    if is_warning_max:
        last_warning_start_max = current_warning_start_max

    # 判断实际是否发生失效
    actual_failure = info.get("out_of_route", False) or info.get('out_of_playfield', False)
    if actual_failure:
        true_label[i] = 1
        # 计算预警步长：选择最早的预警开始时间
        earliest_warning = min(last_warning_start_mean, last_warning_start_max) if last_warning_start_mean > 0 and last_warning_start_max > 0 else max(last_warning_start_mean, last_warning_start_max)
        if earliest_warning > 0:
            steps = cnt - earliest_warning
    
    # 如果没有预警步长记录，steps设为总步数
    if steps == 0:
        steps = cnt
    
    # 计算episode统计
    avg_uncertainty = np.mean(episode_uncertainties)
    max_episode_uncertainty = np.max(episode_uncertainties)
    episode_time = time.time() - start_time
    time_per_step = episode_time / cnt
    
    episode_result = {
        'episode': i,
        'steps': cnt,
        'reward': total_reward,
        'actual_failure': bool(true_label[i]),
        'failure_predicted_mean': bool(pre_label_mean[i]),
        'failure_predicted_max': bool(pre_label_max[i]),
        'warning_start_step_mean': last_warning_start_mean if pre_label_mean[i] == 1 else 0,
        'warning_start_step_max': last_warning_start_max if pre_label_max[i] == 1 else 0,
        'warning_lead_steps': steps,
        'avg_uncertainty': avg_uncertainty,
        'max_uncertainty': max_episode_uncertainty,
        'final_window_mean': uncertainty_mean,
        'final_window_max': uncertainty_max,
        'time': episode_time
    }
    episode_results.append(episode_result)
    
    # 打印结果
    print(f"Episode: {i}\tReward: {total_reward:5.2f}\tPre_Mean: {int(pre_label_mean[i]):2}\t"
          f"Pre_Max: {int(pre_label_max[i]):2}\tTrue: {int(true_label[i]):2}\t"
          f"Unc_Mean: {uncertainty_mean:.4f}\tUnc_Max: {uncertainty_max:.4f}\t"
          f"Steps: {steps:4d}\tT: {time_per_step:.4f}")
    
    # 保存到DataFrame
    df.loc[len(df)] = [i, total_reward, pre_label_mean[i], pre_label_max[i], true_label[i], 
                       uncertainty_mean, uncertainty_max, steps, time_per_step]

# 保存CSV文件
df.to_csv(result_save_dir, index=False)

# 保存不确定性数组
uncertainties_save_path = result_save_dir.replace('.csv', '_uncertainties.npy')
np.save(uncertainties_save_path, np.array(all_uncertainties))

print(f"\nResults saved to: {result_save_dir}")
print(f"Uncertainties saved to: {uncertainties_save_path}")

# 最终分析
print("\n" + "=" * 60)
print("Final Analysis Results (Windowed Uncertainty)")
print("=" * 60)

# 基本统计
total_episodes = len(episode_results)
actual_failures = sum(1 for r in episode_results if r['actual_failure'])
predicted_failures_mean = sum(1 for r in episode_results if r['failure_predicted_mean'])
predicted_failures_max = sum(1 for r in episode_results if r['failure_predicted_max'])

# 预测性能分析 - 基于均值
true_positives_mean = sum(1 for r in episode_results if r['actual_failure'] and r['failure_predicted_mean'])
false_positives_mean = sum(1 for r in episode_results if not r['actual_failure'] and r['failure_predicted_mean'])
true_negatives_mean = sum(1 for r in episode_results if not r['actual_failure'] and not r['failure_predicted_mean'])
false_negatives_mean = sum(1 for r in episode_results if r['actual_failure'] and not r['failure_predicted_mean'])

# 预测性能分析 - 基于最大值
true_positives_max = sum(1 for r in episode_results if r['actual_failure'] and r['failure_predicted_max'])
false_positives_max = sum(1 for r in episode_results if not r['actual_failure'] and r['failure_predicted_max'])
true_negatives_max = sum(1 for r in episode_results if not r['actual_failure'] and not r['failure_predicted_max'])
false_negatives_max = sum(1 for r in episode_results if r['actual_failure'] and not r['failure_predicted_max'])

print(f"Episode Statistics:")
print(f"  Total Episodes: {total_episodes}")
print(f"  Actual Failures: {actual_failures} ({actual_failures/total_episodes:.1%})")
print(f"  Predicted Failures (Mean): {predicted_failures_mean} ({predicted_failures_mean/total_episodes:.1%})")
print(f"  Predicted Failures (Max): {predicted_failures_max} ({predicted_failures_max/total_episodes:.1%})")

print(f"\nPrediction Performance (Window Mean):")
print(f"  True Positives (TP): {true_positives_mean}")
print(f"  False Positives (FP): {false_positives_mean}")
print(f"  True Negatives (TN): {true_negatives_mean}")
print(f"  False Negatives (FN): {false_negatives_mean}")

if (true_positives_mean + false_positives_mean) > 0:
    precision_mean = true_positives_mean / (true_positives_mean + false_positives_mean)
    print(f"  Precision: {precision_mean:.3f}")

if (true_positives_mean + false_negatives_mean) > 0:
    recall_mean = true_positives_mean / (true_positives_mean + false_negatives_mean)
    print(f"  Recall: {recall_mean:.3f}")

if (true_positives_mean + true_negatives_mean) > 0:
    accuracy_mean = (true_positives_mean + true_negatives_mean) / total_episodes
    print(f"  Accuracy: {accuracy_mean:.3f}")

print(f"\nPrediction Performance (Window Max):")
print(f"  True Positives (TP): {true_positives_max}")
print(f"  False Positives (FP): {false_positives_max}")
print(f"  True Negatives (TN): {true_negatives_max}")
print(f"  False Negatives (FN): {false_negatives_max}")

if (true_positives_max + false_positives_max) > 0:
    precision_max = true_positives_max / (true_positives_max + false_positives_max)
    print(f"  Precision: {precision_max:.3f}")

if (true_positives_max + false_negatives_max) > 0:
    recall_max = true_positives_max / (true_positives_max + false_negatives_max)
    print(f"  Recall: {recall_max:.3f}")

if (true_positives_max + true_negatives_max) > 0:
    accuracy_max = (true_positives_max + true_negatives_max) / total_episodes
    print(f"  Accuracy: {accuracy_max:.3f}")

# 预测步长分析
valid_predictions_mean = [r for r in episode_results if r['actual_failure'] and r['failure_predicted_mean'] and r['warning_start_step_mean'] > 0]
valid_predictions_max = [r for r in episode_results if r['actual_failure'] and r['failure_predicted_max'] and r['warning_start_step_max'] > 0]

if len(valid_predictions_mean) > 0:
    lead_steps_mean = [r['warning_lead_steps'] for r in valid_predictions_mean]
    
    print(f"\nPrediction Lead Steps Analysis (Window Mean, Valid: {len(valid_predictions_mean)}):")
    print(f"  Mean Lead Steps: {np.mean(lead_steps_mean):.1f}")
    print(f"  Median Lead Steps: {np.median(lead_steps_mean):.1f}")
    print(f"  Min Lead Steps: {np.min(lead_steps_mean)}")
    print(f"  Max Lead Steps: {np.max(lead_steps_mean)}")
    print(f"  Std Lead Steps: {np.std(lead_steps_mean):.1f}")

if len(valid_predictions_max) > 0:
    lead_steps_max = [r['warning_lead_steps'] for r in valid_predictions_max]
    
    print(f"\nPrediction Lead Steps Analysis (Window Max, Valid: {len(valid_predictions_max)}):")
    print(f"  Mean Lead Steps: {np.mean(lead_steps_max):.1f}")
    print(f"  Median Lead Steps: {np.median(lead_steps_max):.1f}")
    print(f"  Min Lead Steps: {np.min(lead_steps_max)}")
    print(f"  Max Lead Steps: {np.max(lead_steps_max)}")
    print(f"  Std Lead Steps: {np.std(lead_steps_max):.1f}")

# 不确定性分布比较
all_uncertainties = np.array(all_uncertainties)
print(f"\nUncertainty Distribution Comparison:")
print(f"  Training Data - Mean: {np.mean(uncertainty_data):.6f}, Std: {np.std(uncertainty_data):.6f}")
print(f"  Test Data - Mean: {np.mean(all_uncertainties):.6f}, Std: {np.std(all_uncertainties):.6f}")

# 保存完整结果
save_data = {
    'gamma_params': {'shape': shape, 'loc': loc, 'scale': scale},
    'thresholds': thresholds,
    'selected_threshold': selected_threshold,
    'window_size': window_size,
    'training_uncertainties': uncertainty_data,
    'test_uncertainties': all_uncertainties,
    'episode_results': episode_results,
    'statistics': {
        'total_episodes': total_episodes,
        'actual_failures': actual_failures,
        'predicted_failures_mean': predicted_failures_mean,
        'predicted_failures_max': predicted_failures_max,
        'true_positives_mean': true_positives_mean,
        'false_positives_mean': false_positives_mean,
        'true_negatives_mean': true_negatives_mean,
        'false_negatives_mean': false_negatives_mean,
        'true_positives_max': true_positives_max,
        'false_positives_max': false_positives_max,
        'true_negatives_max': true_negatives_max,
        'false_negatives_max': false_negatives_max,
        'valid_predictions_mean': len(valid_predictions_mean) if len(valid_predictions_mean) > 0 else 0,
        'valid_predictions_max': len(valid_predictions_max) if len(valid_predictions_max) > 0 else 0
    },
    'ks_test': {'statistic': ks_statistic, 'p_value': ks_p_value}
}

output_path = './MCD/CarRacing_failure_prediction_results_windowed.npy'
np.save(output_path, save_data)

print(f"\nComplete results saved to: {output_path}")
print(f"Gamma distribution threshold used: {selected_threshold:.8f}")
print(f"Window size: {window_size} steps")