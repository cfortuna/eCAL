from thop import profile
import torch
from torch import nn
from typing import Dict, Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod

class FLOPCalculator(ABC):
    """Abstract class for FLOP calculators should return flops for a single forward pass"""
    @abstractmethod
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        pass

class ThopCalculator(FLOPCalculator):
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        input = torch.randn(input_size)
        forward_flops_per_sample, params = profile(model, inputs=(input,))
        return {
            'total_flops': forward_flops_per_sample
            'total_params': params
        }
