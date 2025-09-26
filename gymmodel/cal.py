import torch
from stable_baselines3 import PPO
import numpy as np

def count_parameters(model):
    """Count the number of parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Calculate model size in MB"""
    total_params = count_parameters(model)
    size_bytes = total_params * 4  # FP32
    size_mb = size_bytes / (1024 * 1024)
    return total_params, size_mb

def analyze_ppo_model_correct(model_path):
    """Correct PPO model parameter analysis"""
    model = PPO.load(model_path)
    
    print("=== Corrected PPO Model Analysis ===")
    print(f"Policy type: {type(model.policy).__name__}")
    
    # Method 1: Total parameters of the entire policy (most accurate)
    total_params, total_size_mb = get_model_size_mb(model.policy)
    print(f"\n📊 Overall Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Total model size: {total_size_mb:.2f} MB")
    
    # Method 2: Analyze main components
    print(f"\n🔍 Main Component Analysis:")
    print(f"{'Component Name':25} | {'Parameters':>10} | {'Size(MB)':>8}")
    print("-" * 55)
    
    # Check the internal structure of shared feature extractor
    if hasattr(model.policy, 'mlp_extractor'):
        extractor = model.policy.mlp_extractor
        
        # Analyze each part of the feature extractor
        if hasattr(extractor, 'shared_net'):
            shared_params, shared_size = get_model_size_mb(extractor.shared_net)
            print(f"{'Shared Network':25} | {shared_params:>8,} | {shared_size:>6.2f}")
        
        if hasattr(extractor, 'policy_net'):
            policy_params, policy_size = get_model_size_mb(extractor.policy_net)
            print(f"{'Policy Feature Network':25} | {policy_params:>8,} | {policy_size:>6.2f}")
            
        if hasattr(extractor, 'value_net'):
            value_params, value_size = get_model_size_mb(extractor.value_net)
            print(f"{'Value Feature Network':25} | {value_params:>8,} | {value_size:>6.2f}")
    
    # Output layers
    if hasattr(model.policy, 'action_net'):
        action_params, action_size = get_model_size_mb(model.policy.action_net)
        print(f"{'Action Output Layer':25} | {action_params:>8,} | {action_size:>6.2f}")
    
    if hasattr(model.policy, 'value_net'):
        value_out_params, value_out_size = get_model_size_mb(model.policy.value_net)
        print(f"{'Value Output Layer':25} | {value_out_params:>8,} | {value_out_size:>6.2f}")
    
    print("-" * 55)
    
    # Method 3: Detailed layer-by-layer analysis
    print(f"\n🏗️ Detailed Layer Analysis:")
    print(f"{'Layer Name':40} | {'Parameters':>10} | {'Size(MB)':>8}")
    print("-" * 70)
    
    layer_count = 0
    for name, module in model.policy.named_modules():
        if len(list(module.children())) == 0:  # Leaf nodes
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                size_mb = params * 4 / (1024 * 1024)
                layer_count += 1
                if layer_count <= 20:  # Only show first 20 layers
                    print(f"{name[:40]:40} | {params:>8,} | {size_mb:>6.2f}")
    
    if layer_count > 20:
        print(f"... ({layer_count-20} more layers)")
    
    print("-" * 70)
    print(f"{'Total Verification':40} | {total_params:>8,} | {total_size_mb:>6.2f}")
    
    # Method 4: Network structure visualization
    print(f"\n🎯 Network Structure:")
    print(model.policy)
    
    return {
        'total_parameters': total_params,
        'total_size_mb': total_size_mb,
        'policy_type': type(model.policy).__name__
    }

# Use the corrected method
model_analysis = analyze_ppo_model_correct("./CarRacing.zip")