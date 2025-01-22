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
        k = 3 # b-spline degree


        raise NotImplementedError("CANCalculator is not implemented")
        return {
            'total_flops': 0,
            'total_params': 0
        }
