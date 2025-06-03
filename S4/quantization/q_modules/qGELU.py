from .base_qmodule import BaseQModule
import torch
import torch.nn as nn

class QGELU(BaseQModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.register_buffer("disable_quantization_flag", torch.tensor(0)) # Register as buffer to store in state_dict


    def quantize(self, weight_quantizer, print_msg=True):
        pass

    def quantize_activations(self, activation_quantizer): # No need to quantize activations within linear layers
        pass

    def disable_quantization(self):
        self.disable_quantization_flag = torch.tensor(1)

    def enable_quantization(self):
        self.disable_quantization_flag = torch.tensor(0)

    def forward(self, x):
        if not self.disable_quantization_flag:
            return x*(torch.minimum(torch.maximum(torch.tensor(0),x+2), torch.tensor(4)) / 4) # Hadamard product with quantised sigmoid
        else:
            return nn.functional.gelu(x)