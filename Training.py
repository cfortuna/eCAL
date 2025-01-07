from thop import profile
import torch
from torch import nn
from typing import Dict, Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod

class FLOPCalculator(ABC):
    """Abstract base class for FLOP calculation strategies"""
    @abstractmethod
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """Calculate FLOPs for a given model"""
        pass

class ThopCalculator(FLOPCalculator):
    """Default FLOP calculator using thop"""
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        input = torch.randn(input_size)
        flops, params = profile(model, inputs=(input,))
        
        return {
            'total_flops': flops,
            'total_params': params
        }

class Training:
    """
    This class is used to estimate the flops of the model training, which is then used to estimate 
    the energy consumption of the model training.
    """
    def __init__(self, calculator: Optional[FLOPCalculator] = None):
        """
        Initialize Training class with optional custom FLOP calculator
        
        Args:
            calculator: Optional custom FLOPCalculator implementation
        """
        self.calculator = calculator if calculator else ThopCalculator()
        
    def set_calculator(self, calculator: FLOPCalculator) -> None:
        """
        Set a new FLOP calculator
        
        Args:
            calculator: FLOPCalculator implementation
        """
        self.calculator = calculator
        
    def calculate_flops(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs using the current calculator
        
        Args:
            model: PyTorch model
            input_size: Input tensor size
            
        Returns:
            Dictionary containing flop calculations
        """
        return self.calculator.calculate(model, input_size)

# Example custom calculator for specialized models(the calculations is probably incorrect this is just an example) 
class TransformerCalculator(FLOPCalculator):
    """Example custom calculator for transformer models"""
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        # Custom implementation for transformer models
        # This is just an example structure
        batch_size, seq_length = input_size[0], input_size[1]
        
        # Example calculation (would need proper implementation)
        attention_flops = self._calculate_attention_flops(model, batch_size, seq_length)
        ffn_flops = self._calculate_ffn_flops(model, batch_size, seq_length)
        
        return {
            'total_flops': attention_flops + ffn_flops,
            'breakdown': {
                'attention': attention_flops,
                'ffn': ffn_flops
            }
        }
    
    def _calculate_attention_flops(self, model, batch_size, seq_length):
        # Implement attention-specific FLOP calculation
        return 0
        
    def _calculate_ffn_flops(self, model, batch_size, seq_length):
        # Implement feed-forward network FLOP calculation
        return 0

# Usage example:
"""
# Default usage with thop
training = Training()
flops = training.calculate_flops(model, input_size)

# Using custom calculator for transformers
transformer_calc = TransformerCalculator()
training = Training(transformer_calc)
# or
training.set_calculator(transformer_calc)
flops = training.calculate_flops(transformer_model, input_size)
"""

