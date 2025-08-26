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
import json
import threading
import subprocess
from datetime import datetime

# GPU监控库导入
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("✓ NVIDIA-ML library available")
except ImportError:
    NVML_AVAILABLE = False
    print("⚠ Warning: pynvml not available, will use alternative monitoring")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"⚠ Warning: pynvml initialization failed: {e}")

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

def safe_float(value):
    """安全地将值转换为float"""
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return 0.0

class GPUMonitor:
    """增强的GPU监控类，支持多种监控方式"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.nvml_available = False
        self.nvidia_smi_available = False
        
        # 尝试初始化NVML
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.nvml_available = True
                print(f"✓ NVML GPU {gpu_id} monitoring enabled")
            except Exception as e:
                print(f"⚠ NVML GPU {gpu_id} access failed: {e}")
        
        # 检查nvidia-smi是否可用
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.nvidia_smi_available = True
                print("✓ nvidia-smi monitoring available as backup")
        except:
            print("⚠ nvidia-smi not available")
    
    def get_gpu_metrics_nvml(self):
        """使用NVML获取GPU指标"""
        if not self.nvml_available:
            return None
            
        try:
            # GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_util = util.gpu
            memory_util = util.memory
            
            # GPU内存使用
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_total = memory_info.total / 1024**2  # MB
            memory_used = memory_info.used / 1024**2    # MB
            memory_free = memory_info.free / 1024**2    # MB
            
            # GPU功耗
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # W
            except:
                power_usage = 0
                
            # GPU温度
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
                
            return {
                'gpu_utilization': gpu_util,
                'memory_utilization': memory_util,
                'memory_total_mb': memory_total,
                'memory_used_mb': memory_used,
                'memory_free_mb': memory_free,
                'power_usage_w': power_usage,
                'temperature_c': temperature,
                'monitoring_method': 'nvml'
            }
        except Exception as e:
            return None
    
    def get_gpu_metrics_nvidia_smi(self):
        """使用nvidia-smi获取GPU指标"""
        if not self.nvidia_smi_available:
            return None
            
        try:
            # 查询GPU信息
            cmd = [
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.total,memory.used,memory.free,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits',
                f'--id={self.gpu_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                
                gpu_util = float(values[0]) if values[0] != '[Not Supported]' else 0
                memory_total = float(values[1]) if values[1] != '[Not Supported]' else 0
                memory_used = float(values[2]) if values[2] != '[Not Supported]' else 0
                memory_free = float(values[3]) if values[3] != '[Not Supported]' else 0
                power_usage = float(values[4]) if values[4] != '[Not Supported]' else 0
                temperature = float(values[5]) if values[5] != '[Not Supported]' else 0
                
                return {
                    'gpu_utilization': gpu_util,
                    'memory_utilization': (memory_used / memory_total * 100) if memory_total > 0 else 0,
                    'memory_total_mb': memory_total,
                    'memory_used_mb': memory_used,
                    'memory_free_mb': memory_free,
                    'power_usage_w': power_usage,
                    'temperature_c': temperature,
                    'monitoring_method': 'nvidia-smi'
                }
        except Exception as e:
            pass
            
        return None
    
    def get_gpu_metrics(self):
        """获取GPU指标，优先使用NVML，备用nvidia-smi"""
        # 尝试NVML
        metrics = self.get_gpu_metrics_nvml()
        if metrics:
            return metrics
        
        # 备用nvidia-smi
        metrics = self.get_gpu_metrics_nvidia_smi()
        if metrics:
            return metrics
        
        # 都失败了，返回默认值
        return {
            'gpu_utilization': 0,
            'memory_utilization': 0,
            'memory_total_mb': 0,
            'memory_used_mb': 0,
            'memory_free_mb': 0,
            'power_usage_w': 0,
            'temperature_c': 0,
            'monitoring_method': 'failed'
        }

class ResourceMonitor:
    """资源监控类，专门为强化学习环境优化"""
    def __init__(self, gpu_id=0, log_interval=10):
        self.gpu_id = gpu_id
        self.log_interval = log_interval
        self.process_pid = os.getpid()
        self.monitoring = False
        self.metrics_history = []
        self.start_time = time.time()
        
        # 初始化GPU监控
        self.gpu_monitor = GPUMonitor(gpu_id)
        
        # 获取进程对象
        self.process = psutil.Process(self.process_pid)
        
        # 碳排放计算器
        self.carbon_tracker = CarbonTracker(region='CN')
        
        # 强化学习特定指标
        self.episode_count = 0
        self.total_reward_sum = 0
        self.inference_times = []
        
        print("✓ RL Resource monitoring initialized")
        
    def get_gpu_metrics(self):
        """获取GPU相关指标"""
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        
        # 获取当前进程的GPU内存使用
        process_gpu_memory = self.get_process_gpu_memory()
        gpu_metrics['process_gpu_memory_mb'] = process_gpu_memory
        
        return gpu_metrics
    
    def get_process_gpu_memory(self):
        """获取当前进程的GPU内存使用"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) == 2:
                            pid, memory = parts
                            if int(pid) == self.process_pid:
                                return float(memory)
        except:
            pass
        
        return 0
    
    def get_cpu_memory_metrics(self):
        """获取CPU和内存指标"""
        try:
            # CPU使用率（当前进程）
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # 内存使用（当前进程）
            memory_info = self.process.memory_info()
            process_memory_mb = memory_info.rss / 1024**2
            process_vms_mb = memory_info.vms / 1024**2
            
            # 系统总体内存
            sys_memory = psutil.virtual_memory()
            sys_memory_percent = sys_memory.percent
            sys_memory_used_mb = sys_memory.used / 1024**2
            sys_memory_total_mb = sys_memory.total / 1024**2
            
            return {
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': process_memory_mb,
                'process_vms_mb': process_vms_mb,
                'system_memory_percent': sys_memory_percent,
                'system_memory_used_mb': sys_memory_used_mb,
                'system_memory_total_mb': sys_memory_total_mb
            }
        except Exception as e:
            return {
                'process_cpu_percent': 0,
                'process_memory_mb': 0,
                'process_vms_mb': 0,
                'system_memory_percent': 0,
                'system_memory_used_mb': 0,
                'system_memory_total_mb': 0
            }
    
    def get_torch_metrics(self):
        """获取PyTorch相关指标"""
        if not torch.cuda.is_available():
            return {
                'torch_allocated_mb': 0,
                'torch_reserved_mb': 0,
                'torch_max_allocated_mb': 0,
                'torch_max_reserved_mb': 0,
                'torch_memory_efficiency': 0
            }
            
        try:
            torch.cuda.synchronize()  # 确保所有操作完成
            
            allocated = torch.cuda.memory_allocated(self.gpu_id)
            reserved = torch.cuda.memory_reserved(self.gpu_id)
            max_allocated = torch.cuda.max_memory_allocated(self.gpu_id)
            max_reserved = torch.cuda.max_memory_reserved(self.gpu_id)
            
            return {
                'torch_allocated_mb': allocated / 1024**2,
                'torch_reserved_mb': reserved / 1024**2,
                'torch_max_allocated_mb': max_allocated / 1024**2,
                'torch_max_reserved_mb': max_reserved / 1024**2,
                'torch_memory_efficiency': allocated / reserved if reserved > 0 else 0
            }
        except Exception as e:
            return {
                'torch_allocated_mb': 0,
                'torch_reserved_mb': 0,
                'torch_max_allocated_mb': 0,
                'torch_max_reserved_mb': 0,
                'torch_memory_efficiency': 0
            }
    
    def collect_metrics(self, episode=None, reward=None, steps=None, phase=None, inference_time=None):
        """收集所有指标"""
        timestamp = time.time()
        elapsed_time = timestamp - self.start_time
        
        # 确保所有值都是可序列化的
        episode = convert_to_serializable(episode)
        reward = convert_to_serializable(reward)
        steps = convert_to_serializable(steps)
        inference_time = convert_to_serializable(inference_time)
        
        metrics = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'phase': phase,
            'inference_time_ms': inference_time
        }
        
        # 收集GPU指标
        gpu_metrics = self.get_gpu_metrics()
        metrics.update(gpu_metrics)
        # 更新碳排放追踪
        self.carbon_tracker.update_energy(gpu_metrics['power_usage_w'], 1)
        
        # 收集CPU/内存指标
        cpu_memory_metrics = self.get_cpu_memory_metrics()
        metrics.update(cpu_memory_metrics)
        
        # 收集PyTorch指标
        torch_metrics = self.get_torch_metrics()
        metrics.update(torch_metrics)
        
        # 更新RL特定统计
        if episode is not None:
            self.episode_count = episode
        if reward is not None:
            self.total_reward_sum += reward
        if inference_time is not None:
            self.inference_times.append(inference_time)
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def start_background_monitoring(self):
        """启动后台监控线程"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("✓ Background monitoring started")
    
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
    
    def save_metrics(self, filepath):
        """保存指标到文件"""
        try:
            # 转换所有数据为可序列化格式
            serializable_metrics = convert_to_serializable(self.metrics_history)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            print(f"✓ Detailed metrics saved to: {filepath}")
            print(f"  Total monitoring samples: {len(self.metrics_history)}")
        except Exception as e:
            print(f"✗ Error saving metrics: {e}")
    
    def save_summary_report(self, filepath):
        """保存汇总报告"""
        summary = self.get_summary_stats()
        if summary:
            try:
                serializable_summary = convert_to_serializable(summary)
                with open(filepath, 'w') as f:
                    json.dump(serializable_summary, f, indent=2)
                print(f"✓ Summary report saved to: {filepath}")
            except Exception as e:
                print(f"✗ Error saving summary: {e}")
    
    def save_human_readable_summary(self, filepath):
        """保存人类可读的汇总报告"""
        summary = self.get_summary_stats()
        if not summary:
            print("No metrics data available for summary")
            return
            
        try:
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write("RL RESOURCE USAGE SUMMARY REPORT\n")
                f.write("="*80 + "\n\n")
                
                exp_info = summary['experiment_info']
                rl_metrics = summary['rl_metrics']
                
                f.write(f"Experiment Duration: {exp_info['total_duration_seconds']:.1f} seconds ({exp_info['total_duration_seconds']/60:.1f} minutes)\n")
                f.write(f"Total Episodes: {rl_metrics['total_episodes']}\n")
                f.write(f"Average Episode Duration: {rl_metrics['avg_episode_duration_seconds']:.2f}s\n")
                f.write(f"Total Monitoring Samples: {exp_info['total_samples']}\n")
                f.write(f"Start Time: {exp_info['start_time']}\n")
                f.write(f"End Time: {exp_info['end_time']}\n\n")
                
                f.write("REINFORCEMENT LEARNING PERFORMANCE:\n")
                f.write(f"  Average Reward: {rl_metrics['average_reward']:.2f}\n")
                f.write(f"  Total Reward Sum: {rl_metrics['total_reward_sum']:.2f}\n")
                f.write(f"  Average Inference Time: {rl_metrics['avg_inference_time_ms']:.2f}ms\n")
                f.write(f"  Max Inference Time: {rl_metrics['max_inference_time_ms']:.2f}ms\n")
                f.write(f"  Inference Throughput: {rl_metrics['inference_fps']:.1f} FPS\n\n")
                
                # 根据GPU监控是否成功显示不同内容
                if summary.get('gpu_metrics') and summary['gpu_metrics']['utilization']['max'] > 0:
                    gpu_metrics = summary['gpu_metrics']
                    f.write("GPU UTILIZATION:\n")
                    f.write(f"  Average: {gpu_metrics['utilization']['avg']:.1f}%\n")
                    f.write(f"  Peak: {gpu_metrics['utilization']['max']:.1f}%\n")
                    f.write(f"  Minimum: {gpu_metrics['utilization']['min']:.1f}%\n")
                    f.write(f"  Standard Deviation: {gpu_metrics['utilization']['std']:.1f}%\n\n")
                    
                    f.write("GPU MEMORY USAGE:\n")
                    f.write(f"  Peak Usage: {gpu_metrics['memory_mb']['peak']:.1f} MB\n")
                    f.write(f"  Average Usage: {gpu_metrics['memory_mb']['avg']:.1f} MB\n")
                    f.write(f"  Minimum Usage: {gpu_metrics['memory_mb']['min']:.1f} MB\n\n")
                else:
                    f.write("GPU MONITORING:\n")
                    f.write("  System-level GPU monitoring failed\n")
                    f.write("  Using PyTorch-level monitoring only\n\n")
                
                # PyTorch内存信息
                torch_metrics = summary['torch_metrics']
                f.write("PYTORCH GPU MEMORY:\n")
                f.write(f"  Peak Allocated: {torch_metrics['max_allocated_mb']:.1f} MB\n")
                f.write(f"  Peak Reserved: {torch_metrics['max_reserved_mb']:.1f} MB\n")
                f.write(f"  Memory Efficiency: {torch_metrics['avg_efficiency']:.3f}\n\n")
                
                # 功耗信息（如果可用）
                power = summary['power_consumption']
                if power['max_watts'] > 0:
                    f.write("POWER CONSUMPTION:\n")
                    f.write(f"  Average Power: {power['avg_watts']:.1f} W\n")
                    f.write(f"  Peak Power: {power['max_watts']:.1f} W\n")
                    f.write(f"  Total Energy: {power['total_energy_wh']:.2f} Wh ({power['total_energy_kwh']:.4f} kWh)\n\n")
                    
                    carbon = summary['carbon_footprint']
                    f.write("ENVIRONMENTAL IMPACT:\n")
                    f.write(f"  Carbon Emission: {carbon['carbon_emission_kg']:.4f} kg CO₂\n")
                    f.write(f"  Equivalent to driving: {carbon['equivalent_car_miles']:.2f} miles\n")
                    f.write(f"  Equivalent to tree absorption: {carbon['equivalent_tree_months']:.1f} tree-months\n\n")
                else:
                    f.write("POWER CONSUMPTION:\n")
                    f.write("  Power monitoring not available\n\n")
                
                cpu_mem = summary['cpu_memory']
                f.write("CPU & MEMORY:\n")
                f.write(f"  Average CPU Usage: {cpu_mem['cpu_avg_percent']:.1f}%\n")
                f.write(f"  Peak CPU Usage: {cpu_mem['cpu_max_percent']:.1f}%\n")
                f.write(f"  Average Memory Usage: {cpu_mem['memory_avg_mb']:.1f} MB\n")
                f.write(f"  Peak Memory Usage: {cpu_mem['memory_peak_mb']:.1f} MB\n\n")
                
                f.write("="*80 + "\n")
                
            print(f"✓ Human-readable summary saved to: {filepath}")
        except Exception as e:
            print(f"✗ Error saving human-readable summary: {e}")
    
    def get_summary_stats(self):
        """获取汇总统计"""
        if not self.metrics_history:
            return None
        
        # 过滤有效数据
        valid_metrics = [m for m in self.metrics_history if m.get('elapsed_time') is not None]
        
        if not valid_metrics:
            return None
        
        total_duration = self.metrics_history[-1]['elapsed_time']
        
        # 提取各类指标
        gpu_utils = [m.get('gpu_utilization', 0) for m in valid_metrics]
        gpu_memory_peaks = [m.get('process_gpu_memory_mb', 0) for m in valid_metrics]
        power_usages = [m.get('power_usage_w', 0) for m in valid_metrics]
        cpu_usages = [m.get('process_cpu_percent', 0) for m in valid_metrics]
        memory_usages = [m.get('process_memory_mb', 0) for m in valid_metrics]
        
        # PyTorch指标
        torch_allocated = [m.get('torch_allocated_mb', 0) for m in valid_metrics]
        torch_reserved = [m.get('torch_reserved_mb', 0) for m in valid_metrics]
        torch_efficiency = [m.get('torch_memory_efficiency', 0) for m in valid_metrics]
        
        # RL特定指标
        episode_rewards = [m.get('reward') for m in valid_metrics if m.get('reward') is not None]
        
        summary = {
            'experiment_info': {
                'total_duration_seconds': total_duration,
                'total_samples': len(valid_metrics),
                'start_time': self.metrics_history[0]['datetime'],
                'end_time': self.metrics_history[-1]['datetime']
            },
            'rl_metrics': {
                'total_episodes': self.episode_count,
                'avg_episode_duration_seconds': total_duration / max(self.episode_count, 1),
                'average_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'total_reward_sum': sum(episode_rewards) if episode_rewards else 0,
                'avg_inference_time_ms': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
                'max_inference_time_ms': max(self.inference_times) if self.inference_times else 0,
                'inference_fps': 1000 / (sum(self.inference_times) / len(self.inference_times)) if self.inference_times else 0
            },
            'torch_metrics': {
                'max_allocated_mb': max(torch_allocated) if torch_allocated else 0,
                'max_reserved_mb': max(torch_reserved) if torch_reserved else 0,
                'avg_efficiency': sum(torch_efficiency) / len(torch_efficiency) if torch_efficiency else 0
            },
            'cpu_memory': {
                'cpu_avg_percent': sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
                'cpu_max_percent': max(cpu_usages) if cpu_usages else 0,
                'memory_avg_mb': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                'memory_peak_mb': max(memory_usages) if memory_usages else 0
            }
        }
        
        # 只有GPU监控成功时才添加GPU指标
        if any(gpu_utils) and max(gpu_utils) > 0:
            summary['gpu_metrics'] = {
                'utilization': {
                    'avg': sum(gpu_utils) / len(gpu_utils),
                    'max': max(gpu_utils),
                    'min': min(gpu_utils),
                    'std': (sum((x - sum(gpu_utils)/len(gpu_utils))**2 for x in gpu_utils) / len(gpu_utils))**0.5 if len(gpu_utils) > 1 else 0
                },
                'memory_mb': {
                    'peak': max(gpu_memory_peaks) if gpu_memory_peaks else 0,
                    'avg': sum(gpu_memory_peaks) / len(gpu_memory_peaks) if gpu_memory_peaks else 0,
                    'min': min(gpu_memory_peaks) if gpu_memory_peaks else 0
                }
            }
        
        # 只有功耗监控成功时才添加功耗指标
        if any(power_usages) and max(power_usages) > 0:
            summary['power_consumption'] = {
                'avg_watts': sum(power_usages) / len(power_usages),
                'max_watts': max(power_usages),
                'total_energy_wh': sum(power_usages) * self.log_interval / 3600,
                'total_energy_kwh': sum(power_usages) * self.log_interval / (3600 * 1000)
            }
            summary['carbon_footprint'] = self.carbon_tracker.get_equivalent_metrics()
        else:
            summary['power_consumption'] = {
                'avg_watts': 0,
                'max_watts': 0,
                'total_energy_wh': 0,
                'total_energy_kwh': 0
            }
            summary['carbon_footprint'] = {
                'total_energy_kwh': 0,
                'carbon_emission_kg': 0,
                'equivalent_car_miles': 0,
                'equivalent_tree_months': 0
            }
        
        return summary
    
    def print_simple_summary(self):
        """打印简单汇总到控制台"""
        summary = self.get_summary_stats()
        if not summary:
            print("No metrics data available for summary")
            return
            
        exp_info = summary['experiment_info']
        rl_metrics = summary['rl_metrics']
        torch_metrics = summary['torch_metrics']
        
        print("\n" + "="*70)
        print("RL RESOURCE USAGE SUMMARY")
        print("="*70)
        print(f"Duration: {exp_info['total_duration_seconds']:.1f}s ({exp_info['total_duration_seconds']/60:.1f}min)")
        print(f"Episodes: {rl_metrics['total_episodes']} | Avg Reward: {rl_metrics['average_reward']:.2f}")
        print(f"Inference: {rl_metrics['avg_inference_time_ms']:.2f}ms avg | {rl_metrics['inference_fps']:.1f} FPS")
        
        if summary.get('gpu_metrics'):
            gpu_metrics = summary['gpu_metrics']
            print(f"GPU Utilization: {gpu_metrics['utilization']['avg']:.1f}% (avg) / {gpu_metrics['utilization']['max']:.1f}% (peak)")
            print(f"GPU Memory: {gpu_metrics['memory_mb']['avg']:.0f}MB (avg) / {gpu_metrics['memory_mb']['peak']:.0f}MB (peak)")
        else:
            print("GPU System Monitoring: Failed")
        
        print(f"PyTorch GPU Memory: {torch_metrics['max_allocated_mb']:.0f}MB (peak) | Efficiency: {torch_metrics['avg_efficiency']:.3f}")
        
        if summary.get('power_consumption') and summary['power_consumption']['max_watts'] > 0:
            power = summary['power_consumption']
            carbon = summary['carbon_footprint']
            print(f"Power: {power['avg_watts']:.1f}W (avg) | Energy: {power['total_energy_wh']:.2f}Wh")
            print(f"Carbon: {carbon['carbon_emission_kg']:.4f}kg CO₂")
        else:
            print("Power Monitoring: Not available")
            
        print("="*70)


class CarbonTracker:
    def __init__(self, region='CN'):
        # 碳强度系数 (g CO2/kWh)
        self.carbon_intensity = {
            'US': 400, 'EU': 300, 'CN': 600, 'FR': 60, 'UK': 250
        }
        self.region = region
        self.total_energy_kwh = 0
        
    def update_energy(self, power_watts, duration_seconds):
        """更新能耗记录"""
        energy_kwh = (power_watts * duration_seconds) / (1000 * 3600)
        self.total_energy_kwh += energy_kwh
        
    def get_carbon_emission(self):
        """计算总碳排放 (kg CO2)"""
        carbon_g = self.total_energy_kwh * self.carbon_intensity[self.region]
        return carbon_g / 1000  # 转换为kg
        
    def get_equivalent_metrics(self):
        """转换为直观的等价指标"""
        carbon_kg = self.get_carbon_emission()
        # 等价对比
        car_miles = carbon_kg / 0.404  # 每英里汽车排放约0.404kg CO2
        tree_months = carbon_kg / (21.77 / 12)  # 一棵树一个月吸收约1.81kg CO2
        
        return {
            'total_energy_kwh': self.total_energy_kwh,
            'carbon_emission_kg': carbon_kg,
            'equivalent_car_miles': car_miles,
            'equivalent_tree_months': tree_months
        }


# 解析参数
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
# 新增监控参数
parser.add_argument('--monitor-interval', default=10, type=int, help='monitoring interval in seconds')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')

# CLI Parse
args = parser.parse_args()

# 初始化资源监控器
monitor = ResourceMonitor(
    gpu_id=args.gpu, 
    log_interval=args.monitor_interval
)

# make dir for exp result
result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

# 监控文件路径
monitor_save_dir = './result/' + args.alg + '/monitoring/'
if not os.path.exists(monitor_save_dir):
    os.makedirs(monitor_save_dir)

metrics_file = monitor_save_dir + args.dataset + '_' + str(args.nsteps) + '_metrics.json'
summary_file = monitor_save_dir + args.dataset + '_' + str(args.nsteps) + '_summary.json'
readable_summary_file = monitor_save_dir + args.dataset + '_' + str(args.nsteps) + '_summary.txt'

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

# 开始后台监控
monitor.start_background_monitoring()
monitor.collect_metrics(phase='initialization')

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
todeynet.load_state_dict(torch.load(model_dir, map_location='cpu'))
todeynet.to(device)
# todeynet.to('cpu')
todeynet.eval()

# 记录模型加载完成
monitor.collect_metrics(phase='model_loaded')

check_episode = args.episodes

# Experiment initialisation
pre_label = np.zeros(check_episode)
true_label = np.zeros(check_episode)

print(f"🚀 Starting RL experiment with {check_episode} episodes...")

# 主要实验循环
for i in range(check_episode):
    episode_start_time = time.time()
    
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
    
    step_inference_times = []
    agent_times = []
    todynet_times = []
    
    while not done and not truncated:
        # RL agent推理时间
        agent_start = time.time()
        action, _ = model.predict(obs, deterministic=True)
        agent_time = (time.time() - agent_start) * 1000  # ms
        agent_times.append(agent_time)
        
        state = torch.as_tensor(obs)
        actions = torch.as_tensor(action)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        rewards = torch.as_tensor([reward], dtype=torch.float32)

        if input_tag == 'SAR':
            record.append(torch.cat([state.view(1, -1)[:, :45].float(), actions.view(1, -1).float(), rewards.view(1, -1).float()], dim=1))
        elif input_tag == 'SA':
            record.append(torch.cat([state.view(1, -1)[:, :45].float(), actions.view(1, -1).float()], dim=1))
        else:
            record.append(torch.cat([state.view(1, -1)[:, :45].float()], dim=1))

        cnt += 1
        
        if len(record) == args.nsteps:
            # TodyNet推理时间

            obs_input = torch.cat(list(record), dim=0).unsqueeze(0)
            obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to(device)
            todynet_start = time.time()
            a = todeynet(obs_input)
            todynet_time = (time.time() - todynet_start) * 1000  # ms
            todynet_times.append(todynet_time)
            step_inference_times.append(agent_time + todynet_time)
            
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
        obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to(device)
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
    
    episode_duration = time.time() - episode_start_time
    t_per_step = episode_duration / cnt
    avg_inference_time = sum(step_inference_times) / len(step_inference_times) if step_inference_times else 0
    
    # 记录episode完成指标
    monitor.collect_metrics(
        episode=i, 
        reward=total_reward, 
        steps=cnt, 
        phase='episode_complete',
        inference_time=avg_inference_time
    )
    avg_agent_time = sum(agent_times) / len(agent_times) if agent_times else 0
    avg_todynet_time = sum(todynet_times) / len(todynet_times) if todynet_times else 0
    
    print(f"Episode: {i:4d}\tReward: {total_reward:7.2f}\tPre: {pre_label[i]:2.0f}\tTrue: {true_label[i]:2.0f}\tProb: {prob:.4f}\tSteps: {steps:4d}\tT: {t_per_step:.4f}\tInf: {avg_inference_time:.2f}ms\tagent_time: {avg_agent_time}\ttodynet_time: {avg_todynet_time}")

    df.loc[len(df)] = [i, total_reward, pre_label[i], true_label[i], prob, steps, t_per_step]
    
    # 每100个episode记录一次资源状态
    if i % 100 == 0 and i > 0:
        monitor.collect_metrics(episode=i, phase=f'checkpoint_{i}')

# 停止监控
monitor.stop_background_monitoring()

# 保存结果
df.to_csv(result_save_dir, index=False)
print(f"✓ Results saved to: {result_save_dir}")

# 打印和保存监控汇总
monitor.print_simple_summary()

# 保存详细监控数据
monitor.save_metrics(metrics_file)
monitor.save_summary_report(summary_file)
monitor.save_human_readable_summary(readable_summary_file)

print(f"\n🎯 Experiment completed!")
print(f"📊 Episodes: {check_episode}")
print(f"📁 Results: {result_save_dir}")
print(f"📈 Monitoring: {monitor_save_dir}")

probs = np.array(probs)