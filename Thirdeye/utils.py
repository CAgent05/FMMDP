import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import gamma
from typing import Union, List, Callable

class Saliency:
    """Vanilla Saliency and Smooth-Grad in PyTorch (单输入版本)"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()  # 将模型设置为评估模式

    def __call__(self,
                 score: Callable,
                 seed_input: torch.Tensor,
                 smooth_samples: int = 0,
                 smooth_noise: float = 0.20,
                 keepdims: bool = False,
                 gradient_modifier: Callable = lambda grads: grads.abs(),
                 normalize_map: bool = True) -> np.ndarray:
        """Generate an attention map that shows how output value changes with respect to a small
        change in input image pixels.

        Args:
            score: A function that takes model outputs and returns a target score.
            seed_input: A torch.Tensor to input into the model.
            smooth_samples: The number of calculating gradients iterations. When over zero,
                           this method will work as SmoothGrad. When zero, it will work as
                           Vanilla Saliency. Defaults to 0.
            smooth_noise: Noise level for SmoothGrad. Defaults to 0.20.
            keepdims: Whether to keep the channel dimension. Defaults to False.
            gradient_modifier: A function to modify gradients. Defaults to taking absolute values.
            normalize_map: Whether to normalize the saliency map. Defaults to True.

        Returns:
            A numpy.ndarray representing the saliency map.
        """
        # 处理 SmoothGrad
        if smooth_samples > 0:
            total_grads = torch.zeros_like(seed_input)
            for _ in range(smooth_samples):
                # 添加噪声
                noise = torch.randn_like(seed_input) * smooth_noise
                noisy_input = seed_input + noise
                # 计算梯度
                grads = self._get_gradients(noisy_input, score, gradient_modifier)
                total_grads += grads
            # 取平均值
            grads = total_grads / smooth_samples
        else:
            # 直接计算梯度
            grads = self._get_gradients(seed_input, score, gradient_modifier)

        # # 转换为 numpy 数组
        # grads = grads.detach().cpu().numpy()

        # # 处理通道维度
        # if not keepdims:
        #     grads = grads.mean(dim=1, keepdim=True)

        # 归一化
        if normalize_map:
            grads = self._normalize(grads)

        grads = grads.squeeze()

        return grads

    def _get_gradients(self,
                       seed_input: torch.Tensor,
                       score: Callable,
                       gradient_modifier: Callable) -> torch.Tensor:
        """Compute gradients of the score with respect to the input."""
        # 启用梯度计算
        seed_input.requires_grad = True

        # 前向传播
        outputs = self.model(seed_input)  # 添加 batch 维度
        score_value = score(outputs)

        # 反向传播
        self.model.zero_grad()
        score_value.backward()

        # 获取梯度
        grads = seed_input.grad

        # 修改梯度
        if gradient_modifier is not None:
            grads = gradient_modifier(grads)

        return grads

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the saliency map."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min == 0:
            return x
        return (x - x_min) / (x_max - x_min)

def get_threshold(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t