from .base_qmodule import BaseQModule
import torch
import torch.nn as nn

class QLinear(BaseQModule):
    def __init__(self, linear_module, **kwargs):
        super().__init__()
        
        self.weight = nn.Parameter(linear_module.weight)
        self.bias = nn.Parameter(linear_module.bias)

        self.weight_quantizer = None
        self.register_buffer("use_quantization_flag", torch.tensor(1)) # Register as buffer to store in state_dict


    def quantize(self, weight_quantizer, print_msg=True):
        if self.weight_quantizer is None:
            self.weight_quantizer = weight_quantizer

        for param_name in weight_quantizer.weight_names:
            # Replace the weights with quantised versions
            assert hasattr(self, param_name), f"Parameter {param_name} not found in qLinear"

            param_to_quantize = getattr(self, param_name).detach()

            # Save original param in case we need it later
            if not hasattr(self, f"orig_{param_name}"):
                self.register_buffer(f"orig_{param_name}", nn.Parameter(param_to_quantize))

            quantizer = getattr(weight_quantizer, param_name)
            quantized_param, scales, zero_points = quantizer(param_to_quantize)

            setattr(self, f"{param_name}", nn.Parameter(quantized_param))
            # self.register_parameter(f"q_{param_name}", nn.Parameter(quantized_param))
            self.register_buffer(f"{param_name}_scales", scales)
            self.register_buffer(f"{param_name}_zero_points", zero_points)

            if print_msg:
                print("Quantized weights for", param_name)


    def quantize_activations(self, activation_quantizer): # No need to quantize activations within linear layers
        pass

    def disable_quantization(self):
        self.use_quantization_flag = torch.tensor(0)

    def enable_quantization(self):
        self.use_quantization_flag = torch.tensor(1)

    def forward(self, x):
        if self.use_quantization_flag:
            weight, bias = self.weight_quantizer.get_quantized_weights(["weight", "bias"], [self.weight, self.bias])
        else:
            weight, bias = getattr(self, "orig_weight", self.weight), getattr(self, "orig_bias", self.bias)
        
        return nn.functional.linear(x, weight, bias)