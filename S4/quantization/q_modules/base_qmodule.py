import torch.nn as nn
from abc import abstractclassmethod

class BaseQModule(nn.Module):
    """
    Base class for quantised modules
    """

    @abstractclassmethod
    def quantize(self, module_weight_quantizer):
        """
        Quantize the module, given the module_weight_quantizer, which defines a quantization scheme for each weight tensor
        """
        raise NotImplementedError
    
    @abstractclassmethod
    def quantize_activations(self, activation_quantizer):
        """
        Quantize the activations, given the activation_quantizer
        activation_quantizer is a partial function
        """
        raise NotImplementedError
    
    @abstractclassmethod
    def disable_quantization(self):
        """
        Disable quantization with the module, i.e. set the module to use full precision
        """
        raise NotImplementedError
    
    @abstractclassmethod
    def enable_quantization(self):
        """
        Enable quantization, i.e. set the module to use quantized precision
        """
        raise NotImplementedError