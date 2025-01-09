from typing import Dict, Union, Tuple
from .model_flops import FLOPCalculator
class Inference:
    """
    This class is used to estimate the flops of the model inference, which is then used to estimate 
    the energy consumption of the model inference.
    """
    def __init__(self, calculator: FLOPCalculator):
        """
        Initialize Inference class
        Args:
            calculator: FLOPCalculator implementation
            model: PyTorch model
            input_size: Tuple of input size
            num_samples: int of number of samples
            processor_flops_per_second: float of processor flops per second
            processor_max_power: int of processor max power in watts
        """
        self.calculator = calculator
        self.model = model
        self.input_size = input_size
        self.num_samples = num_samples
        # hardware parameters
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power


    

    def calculate_flops(self, model: nn.Module, input_size: Tuple, num_samples: int = 1) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs for the current inference
        Args:
            model: PyTorch model
            input_size: Input tensor size
            num_samples: Number of samples to calculate FLOPs for

        Returns:
            Total FLOPs for the current inference   
        """
        forward_flops = self.calculator.calculate(model, input_size)
        total_flops = forward_flops * num_samples
        return total_flops

    def calculate_energy_usage(self) -> float:
        """
        Calculate the energy usage of the current inference

        Returns:
            Total energy usage of the current inference in Joules
        """
        # Calculate the total number of flops
        total_flops = self.calculate_flops()
        # Calculate the total energy usage
        total_time = total_flops / self.processor_flops_per_second
        total_energy = total_time * self.processor_max_power
        return total_energy
    


