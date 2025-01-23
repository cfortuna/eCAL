# from thop import profile
import torch
from torch import nn
from typing import Dict, Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod
from calflops import calculate_flops_hf, calculate_flops


class FlopsCalculatorFactory:
    @staticmethod
    def create_calculator(model: Union[nn.Module, str]) -> 'FLOPCalculator':
        if isinstance(model, str):
            return CalFlopsCalculatorHF()
        elif isinstance(model, nn.Module):
            return CalFlopsCalculatorPT()
        else:
            raise ValueError("Model must be either a string (HuggingFace model name) or nn.Module (PyTorch model)")

class FLOPCalculator(ABC):
    @abstractmethod
    def calculate(self, model: Union[nn.Module, str], input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        pass

class CalFlopsCalculatorHF(FLOPCalculator):
    def calculate(self, model: str, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        flops, macs, params = calculate_flops_hf(model_name=model, input_shape=input_size, print_results=False, output_as_string=False)
        return {"total_flops": flops, "total_params": params}

class CalFlopsCalculatorPT(FLOPCalculator):
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      output_as_string=False,
                                      output_precision=4,
                                      print_results=False)
        return {"total_flops": flops, "total_params": params}


class CANCalculator(FLOPCalculator):
    def __init__(self, grid_size: int,  num_layers : int, din : int, dout : int,  k: int = 3, num_samples: int = 1, num_classes: int = 2):
        self.G = grid_size
        self.din = din
        self.dout = dout
        self.k = k
        self.L = num_layers
        self.T = num_samples
        self.C = num_classes
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        # DO CALCULATION HERE
        """
        Source: https://arxiv.org/pdf/2411.14904v1 section 3
        Nfp = FLOPS of non linear function * T + T * M *[9*k * (G + 1.5*k) + 2 * G -2.5k +3]
         + (L - 2) * (FLOPS of non linear function * M + M^2 * [9k * (G + 1.5k) + 2 * G -2.5k +3])
         + FLOPS of non linear function * M + M *C [9*k*(G + 1.5k) + 2*G - 2.5k + 3]

         din * dout = M * M uniform hidden layer size
         L - number of appropriate univariate nodes(number of layers)
         din, dout - input and output dimensions
         k - b-spline degree fixed to 3
         G - grid size
         T- number of samples?
         C - number of classes

         learnable parameters = (din * dout) *(G + k + 3)+ dout
        """
        k = self.k  # b-spline degree
        G = self.G  
        
        T = self.T
        L = self.L
        C = self.C
        din = self.din
        dout = self.dout
        M = din * dout


        
        # Non-linear function FLOPS (assuming SiLU activation) (x/1+e^(-x)) TODO check this
        nonlinear_flops = 2 

        
        first_term = nonlinear_flops * T + T * M * (9 * k * (G + 1.5 * k) + 2 * G - 2.5 * k + 3)
        middle_term = (L - 2) * (nonlinear_flops * M + M * M * (9 * k * (G + 1.5 * k) + 2 * G - 2.5 * k + 3))
        last_term = nonlinear_flops * M + M * C * (9 * k * (G + 1.5 * k) + 2 * G - 2.5 * k + 3)
        
        total_flops = first_term + middle_term + last_term
        total_params = (din * dout) * (G + k + 3) + dout
        
        return {
            'total_flops': int(total_flops),
            'total_params': int(total_params),
            'breakdown': {
                'input_layer_flops': int(first_term),
                'middle_layers_flops': int(middle_term),
                'output_layer_flops': int(last_term)
            }
        }


        raise NotImplementedError("CANCalculator is not implemented")
        return {
            'total_flops': 0,
            'total_params': 0
        }
