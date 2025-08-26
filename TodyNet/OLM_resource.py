import argparse
import torch
from collections import deque
from utils import prepare_agent
import numpy as np
import pandas as pd
import os
import warnings
import time
import json
import threading
from datetime import datetime
import psutil

warnings.filterwarnings("ignore")

def convert_to_serializable(obj):
    """将不可序列化的对象转换为可序列化格式"""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)

class BaselineResourceMonitor:
    """专门监控强化学习仿真基线资源开销的监控类"""
    
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.metrics_history = []
        self.start_time = time.time()
        
        # 基线测量
        self.baseline_cpu_percent = 0
        self.baseline_memory_mb = 0
        self.baseline_cpu_times = None
        
        # RL仿真统计
        self.episode_count = 0
        self.total_reward_sum = 0
        self.agent_inference_times = []
        
        print("✓ Baseline RL Resource monitoring initialized")
    
    def set_baseline(self):
        """设置系统基线（RL环境启动前）"""
        time.sleep(2)  # 等待系统稳定
        self.baseline_cpu_percent = self.process.cpu_percent(interval=1.0)
        self.baseline_memory_mb = self.process.memory_info().rss / 1024**2
        self.baseline_cpu_times = self.process.cpu_times()
        print(f"✓ System baseline set - CPU: {self.baseline_cpu_percent:.1f}%, Memory: {self.baseline_memory_mb:.1f}MB")
    
    def measure_cpu_usage(self):
        """测量CPU使用情况"""
        try:
            # CPU百分比
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # CPU时间
            cpu_times = self.process.cpu_times()
            user_time = cpu_times.user
            system_time = cpu_times.system
            total_cpu_time = user_time + system_time
            
            # 相对于基线的增量
            if self.baseline_cpu_times:
                user_time_delta = user_time - self.baseline_cpu_times.user
                system_time_delta = system_time - self.baseline_cpu_times.system
                total_cpu_time_delta = user_time_delta + system_time_delta
            else:
                user_time_delta = system_time_delta = total_cpu_time_delta = 0
            
            # 线程和上下文切换
            num_threads = self.process.num_threads()
            ctx_switches = self.process.num_ctx_switches()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_percent_vs_baseline': cpu_percent - self.baseline_cpu_percent,
                'user_cpu_time_seconds': user_time,
                'system_cpu_time_seconds': system_time,
                'total_cpu_time_seconds': total_cpu_time,
                'rl_cpu_time_delta': total_cpu_time_delta,
                'num_threads': num_threads,
                'voluntary_ctx_switches': ctx_switches.voluntary,
                'involuntary_ctx_switches': ctx_switches.involuntary
            }
        except Exception as e:
            print(f"CPU monitoring error: {e}")
            return {}
    
    def measure_memory_usage(self):
        """测量内存使用情况"""
        try:
            memory_info = self.process.memory_info()
            memory_full_info = self.process.memory_full_info()
            
            # 基本内存指标
            rss_mb = memory_info.rss / 1024**2  # 物理内存
            vms_mb = memory_info.vms / 1024**2  # 虚拟内存
            
            # 详细内存指标
            pss_mb = memory_full_info.pss / 1024**2    # 按比例分配的内存
            swap_mb = memory_full_info.swap / 1024**2   # 交换内存
            
            # 系统内存信息
            sys_memory = psutil.virtual_memory()
            
            return {
                'memory_rss_mb': rss_mb,
                'memory_vms_mb': vms_mb,
                'memory_pss_mb': pss_mb,
                'memory_swap_mb': swap_mb,
                'rl_memory_increase_mb': rss_mb - self.baseline_memory_mb,
                'memory_percent_of_system': self.process.memory_percent(),
                'system_memory_available_mb': sys_memory.available / 1024**2,
                'system_memory_usage_percent': sys_memory.percent
            }
        except Exception as e:
            print(f"Memory monitoring error: {e}")
            return {}
    
    def measure_agent_inference(self, inference_func, *args, **kwargs):
        """测量RL智能体推理的资源开销"""
        # 推理前状态
        pre_cpu_times = self.process.cpu_times()
        pre_memory = self.process.memory_info().rss / 1024**2
        
        # 执行推理
        start_time = time.perf_counter()
        result = inference_func(*args, **kwargs)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # 推理后状态
        post_cpu_times = self.process.cpu_times()
        post_memory = self.process.memory_info().rss / 1024**2
        
        # 计算推理开销
        cpu_time_used = (post_cpu_times.user + post_cpu_times.system) - \
                       (pre_cpu_times.user + pre_cpu_times.system)
        memory_delta = post_memory - pre_memory
        
        inference_metrics = {
            'agent_inference_time_ms': inference_time,
            'agent_cpu_time_used_ms': cpu_time_used * 1000,
            'agent_memory_delta_mb': memory_delta,
            'agent_cpu_efficiency': (cpu_time_used * 1000) / inference_time if inference_time > 0 else 0
        }
        
        self.agent_inference_times.append(inference_time)
        
        return result, inference_metrics
    
    def collect_metrics(self, episode=None, reward=None, steps=None, phase=None, custom_data=None):
        """收集RL基线指标"""
        timestamp = time.time()
        elapsed_time = timestamp - self.start_time
        
        metrics = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'episode': convert_to_serializable(episode),
            'reward': convert_to_serializable(reward),
            'steps': convert_to_serializable(steps),
            'phase': phase
        }
        
        # CPU指标
        cpu_metrics = self.measure_cpu_usage()
        metrics.update(cpu_metrics)
        
        # 内存指标
        memory_metrics = self.measure_memory_usage()
        metrics.update(memory_metrics)
        
        # 自定义数据
        if custom_data:
            metrics.update(convert_to_serializable(custom_data))
        
        # 更新RL统计
        if episode is not None:
            self.episode_count = episode
        if reward is not None:
            self.total_reward_sum += reward
        
        self.metrics_history.append(metrics)
        return metrics
    
    def start_background_monitoring(self):
        """启动后台监控线程"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("✓ Background baseline monitoring started")
    
    def stop_background_monitoring(self):
        """停止后台监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("✓ Background monitoring stopped")
    
    def _background_monitor(self):
        """后台监控循环"""
        while self.monitoring:
            self.collect_metrics(phase='background')
            time.sleep(self.log_interval)
    
    def get_summary_stats(self):
        """获取基线汇总统计"""
        if not self.metrics_history:
            return None
        
        valid_metrics = [m for m in self.metrics_history if m.get('elapsed_time') is not None]
        if not valid_metrics:
            return None
        
        # CPU指标统计
        cpu_percents = [m.get('cpu_percent', 0) for m in valid_metrics]
        cpu_deltas = [m.get('rl_cpu_time_delta', 0) for m in valid_metrics]
        
        # 内存指标统计
        memory_rss = [m.get('memory_rss_mb', 0) for m in valid_metrics]
        memory_increases = [m.get('rl_memory_increase_mb', 0) for m in valid_metrics]
        
        # RL特定指标
        episode_rewards = [m.get('reward') for m in valid_metrics if m.get('reward') is not None]
        
        summary = {
            'experiment_info': {
                'total_duration_seconds': self.metrics_history[-1]['elapsed_time'],
                'total_samples': len(valid_metrics),
                'total_episodes': self.episode_count,
                'start_time': self.metrics_history[0]['datetime'],
                'end_time': self.metrics_history[-1]['datetime']
            },
            'rl_performance': {
                'avg_episode_duration_seconds': self.metrics_history[-1]['elapsed_time'] / max(self.episode_count, 1),
                'average_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'total_reward_sum': sum(episode_rewards) if episode_rewards else 0,
                'avg_agent_inference_time_ms': sum(self.agent_inference_times) / len(self.agent_inference_times) if self.agent_inference_times else 0,
                'max_agent_inference_time_ms': max(self.agent_inference_times) if self.agent_inference_times else 0
            },
            'baseline_cpu_analysis': {
                'system_baseline_cpu_percent': self.baseline_cpu_percent,
                'avg_cpu_percent': sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
                'max_cpu_percent': max(cpu_percents) if cpu_percents else 0,
                'rl_cpu_overhead_seconds': max(cpu_deltas) if cpu_deltas else 0,
                'avg_cpu_increase_percent': (sum(cpu_percents) / len(cpu_percents) - self.baseline_cpu_percent) if cpu_percents else 0
            },
            'baseline_memory_analysis': {
                'system_baseline_memory_mb': self.baseline_memory_mb,
                'avg_memory_mb': sum(memory_rss) / len(memory_rss) if memory_rss else 0,
                'max_memory_mb': max(memory_rss) if memory_rss else 0,
                'max_rl_memory_increase_mb': max(memory_increases) if memory_increases else 0,
                'avg_rl_memory_increase_mb': sum(memory_increases) / len(memory_increases) if memory_increases else 0
            }
        }
        
        return summary
    
    def save_baseline_report(self, filepath):
        """保存基线报告"""
        summary = self.get_summary_stats()
        if not summary:
            print("No baseline metrics data available")
            return
            
        try:
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write("REINFORCEMENT LEARNING BASELINE RESOURCE USAGE REPORT\n")
                f.write("="*80 + "\n\n")
                
                exp_info = summary['experiment_info']
                rl_perf = summary['rl_performance']
                cpu_info = summary['baseline_cpu_analysis']
                mem_info = summary['baseline_memory_analysis']
                
                f.write(f"Experiment Duration: {exp_info['total_duration_seconds']:.1f} seconds ({exp_info['total_duration_seconds']/60:.1f} minutes)\n")
                f.write(f"Total Episodes: {exp_info['total_episodes']}\n")
                f.write(f"Average Episode Duration: {rl_perf['avg_episode_duration_seconds']:.2f}s\n")
                f.write(f"Average Reward: {rl_perf['average_reward']:.2f}\n\n")
                
                f.write("RL AGENT PERFORMANCE:\n")
                f.write(f"  Average Agent Inference Time: {rl_perf['avg_agent_inference_time_ms']:.2f}ms\n")
                f.write(f"  Max Agent Inference Time: {rl_perf['max_agent_inference_time_ms']:.2f}ms\n\n")
                
                f.write("BASELINE CPU USAGE:\n")
                f.write(f"  System Baseline CPU: {cpu_info['system_baseline_cpu_percent']:.1f}%\n")
                f.write(f"  Average CPU Usage: {cpu_info['avg_cpu_percent']:.1f}%\n")
                f.write(f"  Peak CPU Usage: {cpu_info['max_cpu_percent']:.1f}%\n")
                f.write(f"  RL CPU Overhead: {cpu_info['rl_cpu_overhead_seconds']:.2f} seconds\n")
                f.write(f"  Average CPU Increase: {cpu_info['avg_cpu_increase_percent']:.1f}%\n\n")
                
                f.write("BASELINE MEMORY USAGE:\n")
                f.write(f"  System Baseline Memory: {mem_info['system_baseline_memory_mb']:.1f} MB\n")
                f.write(f"  Average Memory Usage: {mem_info['avg_memory_mb']:.1f} MB\n")
                f.write(f"  Peak Memory Usage: {mem_info['max_memory_mb']:.1f} MB\n")
                f.write(f"  Max RL Memory Increase: {mem_info['max_rl_memory_increase_mb']:.1f} MB\n")
                f.write(f"  Average RL Memory Increase: {mem_info['avg_rl_memory_increase_mb']:.1f} MB\n\n")
                
                f.write("NOTES:\n")
                f.write("  - This is the BASELINE measurement without TodyNet\n")
                f.write("  - Use this data to calculate TodyNet deployment overhead\n")
                f.write("  - TodyNet overhead = (With TodyNet) - (This Baseline)\n\n")
                
                f.write("="*80 + "\n")
                
            print(f"✓ Baseline report saved to: {filepath}")
        except Exception as e:
            print(f"✗ Error saving baseline report: {e}")
    
    def print_baseline_summary(self):
        """打印基线汇总"""
        summary = self.get_summary_stats()
        if not summary:
            print("No baseline metrics data available")
            return
            
        exp_info = summary['experiment_info']
        rl_perf = summary['rl_performance']
        cpu_info = summary['baseline_cpu_analysis']
        mem_info = summary['baseline_memory_analysis']
        
        print("\n" + "="*70)
        print("RL BASELINE RESOURCE USAGE SUMMARY")
        print("="*70)
        print(f"Duration: {exp_info['total_duration_seconds']:.1f}s | Episodes: {exp_info['total_episodes']}")
        print(f"Avg Reward: {rl_perf['average_reward']:.2f} | Agent Inference: {rl_perf['avg_agent_inference_time_ms']:.2f}ms")
        print(f"CPU: {cpu_info['system_baseline_cpu_percent']:.1f}% → {cpu_info['avg_cpu_percent']:.1f}% (Δ{cpu_info['avg_cpu_increase_percent']:.1f}%)")
        print(f"Memory: {mem_info['system_baseline_memory_mb']:.0f}MB → {mem_info['avg_memory_mb']:.0f}MB (Δ{mem_info['avg_rl_memory_increase_mb']:.1f}MB)")
        print("="*70)

# 解析参数
parser = argparse.ArgumentParser(description='RL Baseline Resource Analysis')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCSA')
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
parser.add_argument('--monitor-interval', default=10, type=int, help='monitoring interval in seconds')

args = parser.parse_args()

# 初始化基线监控器
monitor = BaselineResourceMonitor(log_interval=args.monitor_interval)

# 设置系统基线
print("Setting system baseline...")
monitor.set_baseline()

# 创建结果目录
result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

baseline_save_dir = './result/' + args.alg + '/baseline/'
if not os.path.exists(baseline_save_dir):
    os.makedirs(baseline_save_dir)

baseline_metrics_file = baseline_save_dir + args.dataset + '_baseline_metrics.json'
baseline_report_file = baseline_save_dir + args.dataset + '_baseline_report.txt'
result_save_dir = result_save_dir + '/' + args.dataset + '_baseline.csv'

# 准备环境和智能体
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

# 开始监控
monitor.start_background_monitoring()
monitor.collect_metrics(phase='rl_environment_loaded')

df = pd.DataFrame(columns=['Episode', 'Reward', 'Steps', 'T', 'Agent_Time_ms'])

check_episode = args.episodes

print(f"🚀 Starting RL BASELINE experiment with {check_episode} episodes...")

# 定义智能体推理函数
def agent_inference(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

# 主要实验循环
for i in range(check_episode):
    episode_start_time = time.time()
    
    seed = np.random.randint(5000, 10000)
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0
    cnt = 0
    
    episode_agent_times = []
    
    while not done and not truncated:
        # 使用监控器测量智能体推理开销
        action, inference_metrics = monitor.measure_agent_inference(agent_inference, obs)
        episode_agent_times.append(inference_metrics['agent_inference_time_ms'])
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        cnt += 1
    
    episode_duration = time.time() - episode_start_time
    t_per_step = episode_duration / cnt
    avg_agent_time = sum(episode_agent_times) / len(episode_agent_times) if episode_agent_times else 0
    
    # 记录episode完成指标
    custom_data = {
        'avg_agent_inference_time_ms': avg_agent_time,
        'total_agent_inferences': len(episode_agent_times)
    }
    
    monitor.collect_metrics(
        episode=i, 
        reward=total_reward, 
        steps=cnt, 
        phase='episode_complete',
        custom_data=custom_data
    )
    
    print(f"Episode: {i:4d}\tReward: {total_reward:7.2f}\tSteps: {cnt:4d}\tT: {t_per_step:.4f}\tAgent: {avg_agent_time:.2f}ms")

    df.loc[len(df)] = [i, total_reward, cnt, t_per_step, avg_agent_time]
    
    # 每100个episode记录一次
    if i % 100 == 0 and i > 0:
        monitor.collect_metrics(episode=i, phase=f'checkpoint_{i}')

# 停止监控
monitor.stop_background_monitoring()

# 保存结果
df.to_csv(result_save_dir, index=False)
print(f"✓ Baseline results saved to: {result_save_dir}")

# 打印和保存基线汇总
monitor.print_baseline_summary()

# 保存详细数据
with open(baseline_metrics_file, 'w') as f:
    json.dump(convert_to_serializable(monitor.metrics_history), f, indent=2)

monitor.save_baseline_report(baseline_report_file)

print(f"\n🎯 Baseline experiment completed!")
print(f"📊 Episodes: {check_episode}")
print(f"📁 Results: {result_save_dir}")
print(f"📈 Baseline Analysis: {baseline_report_file}")
print(f"\n💡 Next step: Run with TodyNet and compare!")