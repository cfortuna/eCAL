from .model_flops import FLOPCalculator, ThopCalculator
import torch
from torch import nn
from typing import Dict, Union, Tuple, Callable, Optional

class Training:
    """
    This class is used to estimate the flops of the model training, which is then used to estimate 
    the energy consumption of the model training.
    """
    def __init__(self, model: nn.Module, calculator: Optional[FLOPCalculator] = None, input_size: Tuple,
                 dataset_size: int, batch_size: int, num_epochs: int = 1, num_samples: int,
                 processor_flops_per_second: float = 1e12, processor_max_power: int = 100):
        """
        Initialize Training class with optional custom FLOP calculator
        
        Args:
            calculator: Optional custom FLOPCalculator implementation
            input_size: Tuple of input size
            dataset_size: int of dataset size
            batch_size: int of batch size
            num_epochs: int of number of epochs
            num_samples: int of number of samples
            processor_flops_per_second: float of processor flops per second
            processor_max_power: int of processor max power in watts
        """
        self.calculator = calculator if calculator else ThopCalculator()
        self.model = model
        self.input_size = input_size
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        # hardware parameters
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power
        
    def calculate_flops_training(self) -> Dict[str, Union[int, Dict]]:

        forward_flops = self.calculator.calculate(self.model, self.input_size)
        # 1 training pass takes roughly 3x a single forward pass
        training_flops = forward_flops * 3
        # Calculate the number of batches

        # Calculate the total number of flops
        total_flops = training_flops * self.dataset_size * self.num_epochs
        return total_flops
    def calculate_flops_evaluation(self) -> float:
        # Calculate the total number of flops
        forward_flops = self.calculator.calculate(self.model, self.input_size)
        total_flops = forward_flops * self.num_samples
        return total_flops

    def calculate_energy_usage(self) -> float:
        # Calculate the total number of flops
        training_flops = self.calculate_flops_training()
        evaluation_flops = self.calculate_flops_evaluation()

        # Calculate the total energy usage
        total_flops = training_flops + evaluation_flops
        # Calculate the total energy usage
        running_time = total_flops / self.processor_flops_per_second
        # Calculate the total energy usage
        energy_usage = running_time * self.processor_max_power

        return energy_usage

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

