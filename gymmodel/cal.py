import torch
from stable_baselines3 import PPO
import numpy as np

def count_parameters(model):
    """计算PyTorch模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """计算模型大小(MB)"""
    total_params = count_parameters(model)
    size_bytes = total_params * 4  # FP32
    size_mb = size_bytes / (1024 * 1024)
    return total_params, size_mb

def analyze_ppo_model_correct(model_path):
    """正确的PPO模型参数分析"""
    model = PPO.load(model_path)
    
    print("=== 修正后的PPO模型分析 ===")
    print(f"Policy类型: {type(model.policy).__name__}")
    
    # 方法1: 整个policy的总参数（这是最准确的）
    total_params, total_size_mb = get_model_size_mb(model.policy)
    print(f"\n📊 总体统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"总模型大小: {total_size_mb:.2f} MB")
    
    # 方法2: 分析主要组件
    print(f"\n🔍 主要组件分析:")
    print(f"{'组件名称':25} | {'参数数量':>10} | {'大小(MB)':>8}")
    print("-" * 55)
    
    # 检查共享特征提取器的内部结构
    if hasattr(model.policy, 'mlp_extractor'):
        extractor = model.policy.mlp_extractor
        
        # 分析特征提取器的各个部分
        if hasattr(extractor, 'shared_net'):
            shared_params, shared_size = get_model_size_mb(extractor.shared_net)
            print(f"{'共享网络':25} | {shared_params:>8,} | {shared_size:>6.2f}")
        
        if hasattr(extractor, 'policy_net'):
            policy_params, policy_size = get_model_size_mb(extractor.policy_net)
            print(f"{'策略特征网络':25} | {policy_params:>8,} | {policy_size:>6.2f}")
            
        if hasattr(extractor, 'value_net'):
            value_params, value_size = get_model_size_mb(extractor.value_net)
            print(f"{'价值特征网络':25} | {value_params:>8,} | {value_size:>6.2f}")
    
    # 输出层
    if hasattr(model.policy, 'action_net'):
        action_params, action_size = get_model_size_mb(model.policy.action_net)
        print(f"{'动作输出层':25} | {action_params:>8,} | {action_size:>6.2f}")
    
    if hasattr(model.policy, 'value_net'):
        value_out_params, value_out_size = get_model_size_mb(model.policy.value_net)
        print(f"{'价值输出层':25} | {value_out_params:>8,} | {value_out_size:>6.2f}")
    
    print("-" * 55)
    
    # 方法3: 按层详细分析
    print(f"\n🏗️ 详细层级分析:")
    print(f"{'层名称':40} | {'参数数量':>10} | {'大小(MB)':>8}")
    print("-" * 70)
    
    layer_count = 0
    for name, module in model.policy.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                size_mb = params * 4 / (1024 * 1024)
                layer_count += 1
                if layer_count <= 20:  # 只显示前20层
                    print(f"{name[:40]:40} | {params:>8,} | {size_mb:>6.2f}")
    
    if layer_count > 20:
        print(f"... (还有 {layer_count-20} 层)")
    
    print("-" * 70)
    print(f"{'总计验证':40} | {total_params:>8,} | {total_size_mb:>6.2f}")
    
    # 方法4: 网络结构可视化
    print(f"\n🎯 网络结构:")
    print(model.policy)
    
    return {
        'total_parameters': total_params,
        'total_size_mb': total_size_mb,
        'policy_type': type(model.policy).__name__
    }

# 使用修正后的方法
model_analysis = analyze_ppo_model_correct("/home/cy/PaperWork/FMMDP/gymmodel/CarRacing.zip")