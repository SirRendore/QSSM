import torch
from .tensor_quantizers import TensorQuantizer, tensor_quant, fake_tensor_quant

class ModuleWeightQuantizer():
    '''
    Instantiates a quantizer for each weight tensor in a module
    '''
    def __init__(self, weight_args):
        self.weight_names = list(weight_args.keys())
        for name, weight in weight_args.items():
            setattr(self, name, TensorQuantizer(**weight))

    def get_quantized_weights(self, name_list, tensor_list):
        '''
        Get quantized weights for a list of weight tensors
        '''
        assert len(name_list) == len(tensor_list), "Name list and tensor list must be the same length"
        quantized_list = []
        for name, tensor in zip(name_list, tensor_list):
            try:
                q, _, _ = getattr(self, name)(tensor)
            except AttributeError:
                q = tensor
            quantized_list.append(q)

        return quantized_list
            
    def __repr__(self) -> str:
        # Print weight names and every TensorQuantizer
        print_str = f"ModuleWeightQuantizer with {len(self.weight_names)} weight tensors:\n"
        for name in self.weight_names:
            print_str += f"{name}: {getattr(self, name)}\n"
        return print_str

class ActivationQuantizer(TensorQuantizer):
    '''
    Instantiates a quantizer for input into module
    '''
    def __init__(self, **activation_args):
        self.statistics = None
        self.static_dims_to_reduce = activation_args.get("static_dims_to_reduce", None)
        
        # Remove key if it exists
        try:
            del activation_args["static_dims_to_reduce"]
        except KeyError:
            pass

        super().__init__(**activation_args)

        self.register_buffer("disable_quantization_flag", torch.tensor(0)) # Register as buffer to store in state_dict
        self.register_buffer("use_static_flag", torch.tensor(0)) # Register as buffer to store in state_dict
        # Register amax, amin as buffers
        self.register_buffer("amax", torch.tensor(float("inf")))
        self.register_buffer("amin", torch.tensor(float("inf")))


    def setup_statistics_collection(self):
        self.statistics = None
        self.input_device = None

        def get_statistics(module, input, output):
            if self.input_device is None:
                self.input_device = input[0].device

            # if "input" not in self.statistics:
            #     self.statistics["input"]= input[0].detach().cpu()
            #     self.statistics["output"] = output.detach().cpu()
            # else:
            #     self.statistics["input"] = torch.cat([self.statistics["input"], input[0].detach().cpu()], dim=0)
            #     self.statistics["output"] = torch.cat([self.statistics["output"], output.detach().cpu()], dim=0)
            
            amax_amin = self.get_amax_amin(input[0].detach(), custom_dims_to_reduce=self.static_dims_to_reduce)
            if self.statistics is None:
                if isinstance(amax_amin, tuple):
                    self.register_buffer("amax", amax_amin[0])
                    self.register_buffer("amin", amax_amin[1])
                else:
                    self.register_buffer("amax", amax_amin)

                self.statistics = True
            else:
                # Update amax and amin, comparing to current values
                if isinstance(amax_amin, tuple):
                    self.register_buffer("amax", torch.max(self.amax, amax_amin[0]))
                    self.register_buffer("amin", torch.min(self.amin, amax_amin[1]))
                else:
                    self.register_buffer("amax", torch.max(self.amax, amax_amin))

        
        self.hook_handle = self.register_forward_hook(get_statistics)
        print("Set up collecting statistics for activation quantizer")

    def finish_statistics_collection(self):
        self.hook_handle.remove()
        print("Finished collecting statistics for activation quantizer")
        # self._update_amax_amin()
        self.reset_statistics()
        
    # def _update_amax_amin(self):
    #     if self.statistics is not None:
    #         amax_amin = self.get_amax_amin(self.statistics["input"], custom_dims_to_reduce=self.static_dims_to_reduce)
    #         amax_amin = (amax_amin[0].to(self.input_device), amax_amin[1].to(self.input_device)) if isinstance(amax_amin, tuple) else amax_amin.to(self.input_device)
    #         # Save into buffer
    #         if isinstance(amax_amin, tuple):
    #             self.register_buffer("amax", amax_amin[0])
    #             self.register_buffer("amin", amax_amin[1])
    #         else:
    #             self.register_buffer("amax", amax_amin)
            
    #         print(f"Updated amax and amin for activation quantizer: max_min: {amax_amin}")

    def set_dynamic_quantization(self):
        '''
        Go back to dynamic quantization
        '''
        self.use_static_flag = torch.tensor(0)
        print("Disabled flag for static quantization. Now using dynamic quantization")

    def set_static_quantization(self):
        '''
        Use static quantization
        '''
        # Make sure amax not default value
        if self.amax.shape == torch.tensor(float("inf")).shape and self.amax == torch.tensor(float("inf")):
            raise ValueError("Cannot set static quantization without setting amax/amin. Run calibration first")
        self.use_static_flag = torch.tensor(1)
        print("Enabled flag for static quantization. Now using static quantization")

    def reset_statistics(self):
        self.statistics = None
        print("Reset statistics for activation quantizer")

    def disable_quantization(self):
        self.disable_quantization_flag = torch.tensor(1)
        print("Disabled quantization for activation quantizer")

    def enable_quantization(self):
        self.disable_quantization_flag = torch.tensor(0)
        print("Enabled quantization for activation quantizer")

    def forward(self, inputs):
        if self.disable_quantization_flag: return inputs

        if self.use_static_flag:
            amax_min = (self.amax, self.amin) if self.mode=="asymmetric" else self.amax # Symmetric vs Asymmetric
        else: # Dynamic quantization
            amax_min = self.get_amax_amin(inputs)

        if self.fake_real == "fake":
            output, _, _ =  fake_tensor_quant(inputs, amax_min, self.mode, self.zero_point_bits, self.num_bits, self.unsigned, self.narrow_range)
        elif self.fake_real == "real":
            output, _, _ = tensor_quant(inputs, amax_min, self.mode, self.zero_point_bits, self.num_bits, self.unsigned, self.narrow_range)

        return output

    def __repr__(self) -> str:
        return ("DISABLED - " if self.disable_quantization_flag else "") + \
                f"ActivationQuantizer({self.fake_real=}, {self.num_bits=}, {self.dims_to_reduce=}, {self.percentile=}, {self.mode=}, {self.zero_point_bits=})"