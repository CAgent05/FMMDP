import torch
import torch.nn as nn
from stable_baselines3 import SAC, PPO
import gymnasium as gym

class PPODropoutInjector:
    """直接向已训练PPO模型注入dropout，支持CNN和MLP结构"""
    
    def __init__(self, trained_model_path, cnn_dropout_rate=0.3, mlp_dropout_rate=0.3, env=None):
        # 加载PPO模型
        if env is not None:
            self.model = PPO.load(trained_model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
            print(f"Model loaded on device: {self.model.device}")
            # print("Original model structure:")
            # print(self.model.policy)
        else:
            try:
                self.model = PPO.load(trained_model_path)
            except Exception as e:
                print(f"Standard loading failed: {e}")
                print("Trying alternative loading method...")
                temp_env = gym.make('BipedalWalker-v3')
                self.model = PPO.load(trained_model_path, env=temp_env)
                temp_env.close()
        
        self.cnn_dropout_rate = cnn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        
        # 检测网络类型并注入dropout
        self._inject_dropout_to_networks()
        
        # 默认设为正常推理模式
        self.set_normal_mode()
    
    def _inject_dropout_to_networks(self):
        """向所有网络注入dropout"""
        print("\nAnalyzing PPO policy structure...")
        
        # 检查是否有CNN特征提取器
        if hasattr(self.model.policy, 'features_extractor'):
            print("Found shared features_extractor (CNN)")
            self._add_dropout_to_cnn_extractor(self.model.policy.features_extractor, "features_extractor")
        
        # 处理MLP extractor（如果存在且非空）
        if hasattr(self.model.policy, 'mlp_extractor'):
            print("Processing mlp_extractor...")
            self._add_dropout_to_network(self.model.policy.mlp_extractor, "mlp_extractor")
    
    def _add_dropout_to_cnn_extractor(self, extractor, extractor_name):
        """向CNN特征提取器添加dropout"""
        print(f"  Processing CNN extractor: {extractor_name}")
        
        # 处理CNN部分
        if hasattr(extractor, 'cnn'):
            print(f"    Adding dropout to CNN layers in {extractor_name}")
            self._modify_cnn_with_dropout(extractor.cnn, f"{extractor_name}.cnn")
        
        # 处理线性部分
        if hasattr(extractor, 'linear'):
            print(f"    Adding dropout to linear layers in {extractor_name}")
            self._modify_sequential_with_dropout(extractor.linear, f"{extractor_name}.linear")
    
    def _modify_cnn_with_dropout(self, cnn_sequential, network_name):
        """在CNN Sequential模块中添加dropout - 修复版本"""
        original_layers = list(cnn_sequential.children())
        new_layers = []
        
        print(f"      Original CNN layers: {len(original_layers)}")
        
        i = 0
        while i < len(original_layers):
            current_layer = original_layers[i]
            new_layers.append(current_layer)
            
            # 检查当前层是否是ReLU
            if isinstance(current_layer, nn.ReLU):
                # 添加 Dropout2d，无论是否是最后一个 ReLU
                dropout_layer = nn.Dropout2d(self.cnn_dropout_rate)
                new_layers.append(dropout_layer)
                print(f"        Added Dropout2d after ReLU at position {i}")
            
            i += 1
        
        print(f"      New CNN layers: {len(new_layers)}")
        
        # 重建Sequential模块
        cnn_sequential._modules.clear()
        for idx, layer in enumerate(new_layers):
            cnn_sequential.add_module(str(idx), layer)
    
    def _add_dropout_to_network(self, network, network_name):
        """向网络添加dropout层（通用方法）"""
        for name, module in network.named_children():
            if isinstance(module, nn.Sequential):
                print(f"    Processing {network_name}.{name}")
                self._modify_sequential_with_dropout(module, f"{network_name}.{name}")
            elif hasattr(module, 'named_children'):
                # 递归处理嵌套模块
                self._add_dropout_to_network(module, f"{network_name}.{name}")
    
    def _modify_sequential_with_dropout(self, sequential_module, network_name):
        """在Sequential模块的激活函数后添加dropout"""
        original_layers = list(sequential_module.children())
        
        # 如果Sequential为空，跳过
        if len(original_layers) == 0:
            print(f"      {network_name} is empty, skipping")
            return
        
        new_layers = []
        
        for i, layer in enumerate(original_layers):
            new_layers.append(layer)
            
            # 在激活函数后添加dropout（但不在最后一层）
            if isinstance(layer, (nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.ELU)) and i < len(original_layers) - 1:
                dropout_layer = nn.Dropout(self.mlp_dropout_rate)
                new_layers.append(dropout_layer)
                print(f"      Added Dropout after {type(layer).__name__} in {network_name}")
        
        # 重建Sequential模块
        sequential_module._modules.clear()
        for idx, layer in enumerate(new_layers):
            sequential_module.add_module(str(idx), layer)
    
    def set_normal_mode(self):
        """设置为正常推理模式（不使用dropout）"""
        self.model.policy.set_training_mode(False)
        for module in self.model.policy.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.eval()
        # print("Mode: NORMAL (dropout disabled)")
    
    def set_dropout_mode(self):
        """设置为dropout模式（启用dropout进行不确定性估计）"""
        self.model.policy.set_training_mode(True)
        for module in self.model.policy.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
        # print("Mode: DROPOUT (dropout enabled)")
    
    def predict(self, observation, deterministic=True):
        """
        统一的预测接口，根据当前模式进行推理
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def predict_value(self, observation):
        """预测状态价值"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            value = self.model.policy.predict_values(obs_tensor)
            return value.cpu().numpy().item()
    
    def check_dropout_injection(self):
        """检查dropout是否成功注入"""
        print("\n=== Checking PPO Dropout Injection ===")
        dropout_count = 0
        dropout2d_count = 0
        
        for name, module in self.model.policy.named_modules():
            if isinstance(module, nn.Dropout):
                print(f"✓ Found Dropout at: {name} (rate: {module.p})")
                dropout_count += 1
            elif isinstance(module, nn.Dropout2d):
                print(f"✓ Found Dropout2d at: {name} (rate: {module.p})")
                dropout2d_count += 1
        
        total_dropout = dropout_count + dropout2d_count
        print(f"Total Dropout layers: {dropout_count}")
        print(f"Total Dropout2d layers: {dropout2d_count}")
        print(f"Total dropout layers: {total_dropout}")
        return total_dropout > 0
    
    def print_network_structure(self):
        """打印网络结构"""
        print("\n=== PPO Network Structure After Dropout Injection ===")
        
        # CNN特征提取器
        if hasattr(self.model.policy, 'features_extractor'):
            print("\nShared Features Extractor structure:")
            for name, module in self.model.policy.features_extractor.named_modules():
                if name:
                    print(f"  {name}: {type(module).__name__}")
        
        if hasattr(self.model.policy, 'pi_features_extractor'):
            print("\nPolicy Features Extractor structure:")
            for name, module in self.model.policy.pi_features_extractor.named_modules():
                if name:
                    print(f"  {name}: {type(module).__name__}")
        
        if hasattr(self.model.policy, 'vf_features_extractor'):
            print("\nValue Features Extractor structure:")
            for name, module in self.model.policy.vf_features_extractor.named_modules():
                if name:
                    print(f"  {name}: {type(module).__name__}")