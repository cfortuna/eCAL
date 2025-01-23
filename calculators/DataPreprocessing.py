from typing import Dict, Union
from .PreprocessingFLOPS import *

class DataPreprocessing:
    """Data preprocessing class that calculates the FLOPs for various data preprocessing tasks"""
    
    def __init__(self, preprocessing_type: str = 'normalization', processor_flops_per_second: float = 1e12, processor_max_power: int = 100, time_steps: int = 1):
        """
        Initialize DataPreprocessing class
        
        Args:
            preprocessing_type: Type of preprocessing to perform
        """
        self.calculators = {
            'normalization': NormalizationCalculator(),
            'min_max_scaling': MinMaxScalingCalculator(),
            'GADF': GramianDifferenceFieldCalculator(time_steps)
        }
        self.set_preprocessing_type(preprocessing_type)
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power
    
    def set_preprocessing_type(self, preprocessing_type: str) -> None:
        """Set the preprocessing type"""
        if preprocessing_type not in self.calculators:
            raise ValueError(f"Unsupported preprocessing type: {preprocessing_type}")
        self.calculator = self.calculators[preprocessing_type]
    
    def calculate_flops(self, data_bits: int) -> float:
        """
        Calculate FLOPs for the current preprocessing type

        Args:
            data_size: Size of the data to preprocess
        
        Returns:
            Total FLOPs for the current preprocessing type
        """
        return self.calculator.calculate_flops(data_bits)

    def calculate_energy(self, data_bits: int) -> float:
        # Calculate the total number of flops
        calc_dict = self.calculate_flops(data_bits)
        total_flops = calc_dict['total_flops']
        # data_shape = calc_dict['data_shape']
        # if data_shape is not None:
        #     #TODO implement handling for GADF
        #     raise NotImplementedError("GADF handling is not implemented")
        # Calculate the total energy usage
        total_time = total_flops / self.processor_flops_per_second
        total_energy = total_time * self.processor_max_power
        return {"total_energy": total_energy}

