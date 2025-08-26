import argparse
import time
import gc
import random
import os
import json
import threading
from datetime import datetime
import subprocess

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

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

from net import GNNStack
from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader
import warnings

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
            print(f"NVML error: {e}")
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
            print(f"nvidia-smi error: {e}")
            
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

# 资源监控类
class ResourceMonitor:
    def __init__(self, gpu_id=0, log_interval=5):
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
        
        print("✓ Resource monitoring initialized")
        
    def get_gpu_metrics(self):
        """获取GPU相关指标"""
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        
        # 获取当前进程的GPU内存使用（通过nvidia-smi）
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
    
    def collect_metrics(self, step=None, epoch=None, phase=None, loss=None, acc=None):
        """收集所有指标"""
        timestamp = time.time()
        elapsed_time = timestamp - self.start_time
        
        # 确保所有值都是可序列化的
        step = convert_to_serializable(step)
        epoch = convert_to_serializable(epoch)
        loss = convert_to_serializable(loss)
        acc = convert_to_serializable(acc)
        
        metrics = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'step': step,
            'epoch': epoch,
            'phase': phase,
            'loss': loss,
            'accuracy': acc
        }
        
        # 收集GPU指标
        gpu_metrics = self.get_gpu_metrics()
        metrics.update(gpu_metrics)
        # 更新碳排放追踪
        self.carbon_tracker.update_energy(gpu_metrics['power_usage_w'], 1)  # 假设1秒间隔
        
        # 收集CPU/内存指标
        cpu_memory_metrics = self.get_cpu_memory_metrics()
        metrics.update(cpu_memory_metrics)
        
        # 收集PyTorch指标
        torch_metrics = self.get_torch_metrics()
        metrics.update(torch_metrics)
        
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
                f.write("RESOURCE USAGE SUMMARY REPORT\n")
                f.write("="*80 + "\n\n")
                
                exp_info = summary['experiment_info']
                
                f.write(f"Experiment Duration: {exp_info['total_duration_seconds']:.1f} seconds ({exp_info['total_duration_seconds']/60:.1f} minutes)\n")
                f.write(f"Total Monitoring Samples: {exp_info['total_samples']}\n")
                f.write(f"Start Time: {exp_info['start_time']}\n")
                f.write(f"End Time: {exp_info['end_time']}\n\n")
                
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
        
        summary = {
            'experiment_info': {
                'total_duration_seconds': total_duration,
                'total_samples': len(valid_metrics),
                'start_time': self.metrics_history[0]['datetime'],
                'end_time': self.metrics_history[-1]['datetime']
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
        torch_metrics = summary['torch_metrics']
        cpu_mem = summary['cpu_memory']
        
        print("\n" + "="*60)
        print("RESOURCE USAGE SUMMARY")
        print("="*60)
        print(f"Duration: {exp_info['total_duration_seconds']:.1f}s ({exp_info['total_duration_seconds']/60:.1f}min)")
        
        if summary.get('gpu_metrics'):
            gpu_metrics = summary['gpu_metrics']
            print(f"GPU Utilization: {gpu_metrics['utilization']['avg']:.1f}% (avg) / {gpu_metrics['utilization']['max']:.1f}% (peak)")
            print(f"GPU Memory: {gpu_metrics['memory_mb']['avg']:.0f}MB (avg) / {gpu_metrics['memory_mb']['peak']:.0f}MB (peak)")
        else:
            print("GPU System Monitoring: Failed")
        
        print(f"PyTorch GPU Memory: {torch_metrics['max_allocated_mb']:.0f}MB (peak allocated) / {torch_metrics['max_reserved_mb']:.0f}MB (peak reserved)")
        print(f"Memory Efficiency: {torch_metrics['avg_efficiency']:.3f}")
        print(f"CPU: {cpu_mem['cpu_max_percent']:.1f}% (peak) / RAM: {cpu_mem['memory_peak_mb']:.0f}MB (peak)")
        
        if summary.get('power_consumption') and summary['power_consumption']['max_watts'] > 0:
            power = summary['power_consumption']
            carbon = summary['carbon_footprint']
            print(f"Power: {power['avg_watts']:.1f}W (avg) / {power['total_energy_wh']:.2f}Wh (total)")
            print(f"Carbon: {carbon['carbon_emission_kg']:.4f}kg CO₂ ≈ {carbon['equivalent_car_miles']:.2f} car miles")
        else:
            print("Power Monitoring: Not available")
            
        print("="*60)


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


class ModelComplexityAnalyzer:
    @staticmethod
    def count_parameters(model):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def estimate_model_size(model):
        """估算模型大小"""
        total_params, _ = ModelComplexityAnalyzer.count_parameters(model)
        # 假设每个参数是float32 (4 bytes)
        model_size_mb = total_params * 4 / 1024**2
        return model_size_mb


# 修改原始代码
torch.cuda.set_device(0)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='StandWalkJump')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', 
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=256, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# 新增监控相关参数
parser.add_argument('--monitor-interval', default=5, type=int,
                    help='monitoring interval in seconds')
parser.add_argument('--tag', default='exp', type=str,
                    help='experiment tag')


def main():
    args = parser.parse_args()
    
    args.kern_size = [ int(l) for l in args.kern_size.split(",") ]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_work(args)


def main_work(args):
    # 初始化资源监控器
    monitor = ResourceMonitor(
        gpu_id=args.gpu, 
        log_interval=args.monitor_interval
    )
    
    # init acc
    best_acc1 = 0
    
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = './TodyNet/log/{}_gpu{}_{}_{}_exp.txt'.format(args.tag, args.gpu, args.arch, args.dataset)
    metrics_file = './TodyNet/log/{}_gpu{}_{}_{}_metrics.json'.format(args.tag, args.gpu, args.arch, args.dataset)
    summary_file = './TodyNet/log/{}_gpu{}_{}_{}_summary.json'.format(args.tag, args.gpu, args.arch, args.dataset)
    readable_summary_file = './TodyNet/log/{}_gpu{}_{}_{}_summary.txt'.format(args.tag, args.gpu, args.arch, args.dataset)
    
    if not os.path.exists('./TodyNet/log/'):
        os.makedirs('./TodyNet/log/')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # 开始后台监控
    monitor.start_background_monitoring()
    
    # dataset
    train_loader, val_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(args)
    
    # training model from net.py
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers, 
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size, 
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, 
                     seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes)

    # 分析模型复杂度
    total_params, trainable_params = ModelComplexityAnalyzer.count_parameters(model)
    model_size_mb = ModelComplexityAnalyzer.estimate_model_size(model)
    
    # print & log
    log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
    log_msg('num_nodes {}, seq_length {}, num_classes {}'.format(num_nodes, seq_length, num_classes), log_file)
    log_msg('Model: total_params {}, trainable_params {}, size {:.2f}MB'.format(
        total_params, trainable_params, model_size_mb), log_file)
    log_msg('date {}, gpu {}, arch{}'.format(args.tag, args.gpu, args.arch), log_file)

    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Warning! Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

        # collect cache
        gc.collect()
        torch.cuda.empty_cache()

        model = model.cuda(args.gpu)
        # 记录模型加载后的指标
        monitor.collect_metrics(phase='model_loaded')
        
        if hasattr(args, 'use_benchmark') and args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error! We only have one gpu!!!")

    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                              patience=50, verbose=True)

    # validation
    if args.evaluate:
        acc_train_per, loss_train_per = validate(val_loader, model, criterion, args, monitor)
        print(acc_train_per, loss_train_per)

        msg = f'loss {loss_train_per}, acc {acc_train_per}'
        log_msg(msg, log_file)
        
        # 停止监控并保存指标
        monitor.stop_background_monitoring()
        monitor.print_simple_summary()
        
        # 保存所有文件
        monitor.save_metrics(metrics_file)
        monitor.save_summary_report(summary_file)
        monitor.save_human_readable_summary(readable_summary_file)
        return

    # train & valid
    print('****************************************************')
    print(f"Starting training on dataset: {args.dataset}")

    dataset_time = AverageMeter('Time', ':6.3f')

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoches = []

    end = time.time()
    for epoch in range(args.epochs):
        epoches += [epoch]

        # train for one epoch
        acc_train_per, loss_train_per = train(train_loader, model, criterion, optimizer, 
                                            lr_scheduler, args, monitor, epoch)
        
        acc_train += [acc_train_per]
        loss_train += [loss_train_per]

        msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per}'
        log_msg(msg, log_file)

        # evaluate on validation set
        acc_val_per, loss_val_per = validate(val_loader, model, criterion, args, monitor, epoch)

        acc_val += [acc_val_per]
        loss_val += [loss_val_per]

        msg = f'VAL, loss {loss_val_per}, acc {acc_val_per}'
        log_msg(msg, log_file)

        # remember best acc - 安全地处理tensor
        best_acc1 = max(safe_float(acc_val_per), safe_float(best_acc1))

    # measure elapsed time
    dataset_time.update(time.time() - end)

    # 停止监控
    monitor.stop_background_monitoring()
    
    # 打印简单汇总
    monitor.print_simple_summary()
    
    # 获取资源使用汇总
    summary = monitor.get_summary_stats()
    
    # log & print the best_acc - 安全地处理tensor
    best_acc1_safe = safe_float(best_acc1)
    msg = f'\n\n * BEST_ACC: {best_acc1_safe}\n * TIME: {dataset_time}\n'
    if summary:
        msg += f' * RESOURCE SUMMARY:\n'
        msg += f'   - Total Duration: {summary["experiment_info"]["total_duration_seconds"]:.2f}s\n'
        if summary.get('gpu_metrics'):
            msg += f'   - Avg GPU Utilization: {summary["gpu_metrics"]["utilization"]["avg"]:.1f}%\n'
            msg += f'   - Peak GPU Memory: {summary["gpu_metrics"]["memory_mb"]["peak"]:.1f}MB\n'
        msg += f'   - PyTorch Peak GPU Memory: {summary["torch_metrics"]["max_allocated_mb"]:.1f}MB\n'
        if summary.get('power_consumption') and summary['power_consumption']['max_watts'] > 0:
            msg += f'   - Total Energy: {summary["power_consumption"]["total_energy_wh"]:.2f}Wh\n'
            msg += f'   - Carbon Emission: {summary["carbon_footprint"]["carbon_emission_kg"]:.4f}kg CO2\n'
            msg += f'   - Equivalent Car Miles: {summary["carbon_footprint"]["equivalent_car_miles"]:.2f}\n'
    
    log_msg(msg, log_file)

    print(f'\n🎯 Final Results:')
    print(f' * Best Accuracy: {best_acc1_safe:.2f}%')
    print(f' * Training Time: {dataset_time}')
    
    # 保存模型
    save_dir = './model/Todynet/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    weights_file = './model/Todynet/{}_{}.pth'.format(args.dataset, args.nsteps)
    torch.save(model.state_dict(), weights_file)
    print(f"✓ Model saved to: {weights_file}")

    # 保存详细指标
    monitor.save_metrics(metrics_file)
    monitor.save_summary_report(summary_file)
    monitor.save_human_readable_summary(readable_summary_file)

    # collect cache
    gc.collect()
    torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, lr_scheduler, args, monitor, epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    # switch to train mode
    model.train()
    
    # 记录epoch开始
    monitor.collect_metrics(epoch=epoch, phase='train_start')

    for count, (data, label) in enumerate(train_loader):
        # 记录关键batch的资源状态
        if count % 50 == 0:  # 每50个batch记录一次
            monitor.collect_metrics(step=count, epoch=epoch, phase='train_batch')

        # data in cuda
        data = data.cuda(args.gpu).type(torch.float)
        label = label.cuda(args.gpu).type(torch.long)

        # compute output
        output = model(data)
    
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1 = accuracy(output, label, topk=(1, 1))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    lr_scheduler.step(top1.avg)

    # 记录epoch结束时的状态
    monitor.collect_metrics(epoch=epoch, phase='train_end', 
                          loss=losses.avg, acc=top1.avg)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, monitor, epoch=None):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    # 记录验证开始
    monitor.collect_metrics(epoch=epoch, phase='val_start')

    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True).type(torch.float)
            if torch.cuda.is_available():
                label = label.cuda(args.gpu, non_blocking=True).type(torch.long)
            # compute output
            output = model(data)

            loss = criterion(output, label)

            # measure accuracy and record loss
            acc1 = accuracy(output, label, topk=(1, 1))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

    # 记录验证结束
    monitor.collect_metrics(epoch=epoch, phase='val_end', 
                          loss=losses.avg, acc=top1.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()