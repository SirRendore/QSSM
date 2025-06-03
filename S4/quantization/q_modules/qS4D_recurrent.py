# S4D Recurrent Mode
from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
from .base_qmodule import BaseQModule

def c2r(mode):
    '''
    Given a mode, return the real to complex and complex to real functions
    '''
    if mode == "cartesian":
        return torch.view_as_real, torch.view_as_complex
    elif mode == "polar":
        return lambda x: torch.stack((torch.abs(x), torch.angle(x)), dim=-1), lambda x: torch.polar(x[..., 0], x[..., 1])
    else:
        raise ValueError(f"Invalid real to complex mode {mode}")


class S4BlockRecurrent(BaseQModule):
    """General recurrent block for S4D model

    Args:
        s4_block (S4Block): S4Block to be replaced
        qdtype (torch.dtype): Quantisation type
    """

    def __init__(
        self,
        s4_block,
        **kwargs
    ):
        super().__init__()

        self.d_model = s4_block.d_model
        self.transposed = s4_block.transposed

        self.gate = s4_block.gate
        self.bottleneck = s4_block.bottleneck

        if self.bottleneck is not None:
            self.input_linear = s4_block.input_linear

        if self.gate is not None:
            self.input_gate = s4_block.input_gate
            if s4_block.layer.d_output != self.d_model * self.gate:
                self.output_gate = s4_block.output_gate

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        # self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)
                
        self.layer_activation = s4_block.layer.activation

        # State matrices
        dt, A, B, C, = s4_block.layer.kernel._get_params()
        dtA = dt * A  # (H N)
        if s4_block.layer.kernel.disc == 'zoh':
            dA = torch.exp(dtA) # (H N)
            dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif s4_block.layer.kernel.disc == 'bilinear':
            dA = (1. + dtA/2) / (1. - dtA/2)
            dB = B * (1. - dtA/2).reciprocal() * dt # or * dtA / A
        dB = rearrange(dB, '1 h n -> h n')
        dC = C

        self.dA = nn.Parameter(dA) # (H N//2)
        self.dB = nn.Parameter(dB) # (H N//2)
        self.dC = nn.Parameter(dC)
        self.D = nn.Parameter(s4_block.layer.D) # (H)
        self.N = s4_block.layer.kernel.N

        self.parameterized_A = False # Flag to check if A matrix is parameterized

        # Quantisation params ------------
        self.state_quantizer = None
        self.y_recur_quantizer = None
        self.y_pre_output_quantizer = None

        self.weight_quantizer = None
        self.register_buffer("disable_quantization_flag", torch.tensor(0)) # Register as buffer to store in state_dict


        # Real to complex conversion
        self.complex_to_real_mode = kwargs.get("complex_to_real_mode", "cartesian")
        self.complex_to_real, self.real_to_complex = c2r(self.complex_to_real_mode)

        # Pointwise operation
        # Activation after (optional) multiplication by gate branch
        self.mult_activation = s4_block.mult_activation
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        self.drop = s4_block.drop

        self.output_linear = s4_block.output_linear

        # Clamping limits
        self.state_clamp_max = kwargs.get("state_clamp_max", 50) # Default is 50, can overwrite in config
        self.state_clamp_min = kwargs.get("state_clamp_min", -50)
        self.y_recur_clamp_max = kwargs.get("y_recur_clamp_max", 1e3)
        self.y_recur_clamp_min = kwargs.get("y_recur_clamp_min", -1e3)
        
        # Gradient clipping on state quantizer
        self.state_quantizer_clip = kwargs.get("state_quantizer_clip", None)
        # Dimensions to reduce for state quantizer
        self.state_quantizer_dims = kwargs.get("state_quantizer_dims", [-1, -2])
        # Bits for state quantizer
        self.state_quantizer_bits = kwargs.get("state_quantizer_bits", None)

    def enable_parameterized_A(self):
        """
        Parameterize A matrix for S4BlockRecurrent

        Returns:
            torch.Tensor: Parameterized A matrix
        """
        # Magnitude
        self.A_mag = nn.Parameter(torch.abs(self.dA))
        # Theta
        self.A_theta = nn.Parameter(torch.angle(self.dA))
        # Return A
        self.dA_inv_fn = partial(torch.polar, self.A_mag, self.A_theta)

        print("Parameterized A matrix for S4BlockRecurrent")
        self.parameterized_A = True

    def disable_parameterized_A(self):
        """
        Disable parameterization of A matrix for S4BlockRecurrent
        """
        self.parameterized_A = False
        print("Disabled parameterization of A matrix for S4BlockRecurrent")


    def update_parameterized_A(self):
        if hasattr(self, "A_mag") and hasattr(self, "A_theta"):
            self.dA = nn.Parameter(self.dA_inv_fn())
            del self.A_mag, self.A_theta

    
    def quantize_activations(self, act_quantizer):
        """
        Quantizes the activations of the block

        Args:
            act_quantizer (ActivationQuantizer): Activation quantizer object, a partial function
        """
        if act_quantizer is not None:
            # Custom quantizers for activations
            if self.state_quantizer_bits is not None:
                self.state_quantizer = act_quantizer(num_bits=self.state_quantizer_bits, dims_to_reduce = self.state_quantizer_dims)
            else:
                self.state_quantizer = act_quantizer(dims_to_reduce = self.state_quantizer_dims) # (B H N C=2) per channel
            self.y_recur_quantizer = act_quantizer(dims_to_reduce = [-2]) # # (B H L) per tensor
            self.y_pre_output_quantizer = act_quantizer(dims_to_reduce=[-2]) # (B H L) 
            print("Quantized activations inside S4BlockRecurrent")
            
            # Register hook on backward of state quantizer if clipping set
            if self.state_quantizer_clip is not None:
                self.state_quantizer_handle = self.state_quantizer.register_full_backward_hook(lambda module, grad_input, grad_output: (torch.clamp(grad_input[0], min=-self.state_quantizer_clip, max=self.state_quantizer_clip), ))
                print("Registered clipping hook on state quantizer")
        else:
            print("No activation quantizer provided for S4BlockRecurrent")
        

    def disable_quantization(self):
        """
        Disables quantization for the block

        """
        self.disable_quantization_flag = torch.tensor(1)
        # self.state_quantizer.disable_quantization()
        # self.y_recur_quantizer.disable_quantization()
        # self.y_pre_output_quantizer.disable_quantization()
        print(f"Disabled quantization for S4BlockRecurrent")

    def enable_quantization(self):
        '''
        Enables quantization for the block
        '''
        self.disable_quantization_flag = torch.tensor(0)
        # self.state_quantizer.enable_quantization()
        # self.y_recur_quantizer.enable_quantization()
        # self.y_pre_output_quantizer.enable_quantization()
        print("Enabled quantization for S4BlockRecurrent")
        

    def quantize(self, weight_quantizer, print_msg=True):
        """
        Quantizes the weights of the block

        Args:
            weight_quantizer (ModuleWeightQuantizer): Weight quantizer object
            print_msg (bool): Print message if True
        """
       
        if self.weight_quantizer is None:
            self.weight_quantizer = weight_quantizer
        
        for param_name in weight_quantizer.weight_names:
            # Replace the weights with quantised versions
            assert hasattr(self, param_name), f"Parameter {param_name} not found in S4BlockRecurrent"

            param_to_quantize = getattr(self, param_name).detach()

            # Save original param in case we need it later
            if not hasattr(self, f"orig_{param_name}"):
                self.register_buffer(f"orig_{param_name}", nn.Parameter(param_to_quantize))

            # If the parameter is complex, we need to view it as real to quantize
            if param_name in ['dA', 'dB', 'dC']:
                param_to_quantize = self.complex_to_real(param_to_quantize)

            quantizer = getattr(weight_quantizer, param_name)
            quantized_param, scales, zero_points = quantizer(param_to_quantize)

            # If the original parameter is complex, we need to view it as complex again
            if param_name in ['dA', 'dB', 'dC']:
                quantized_param = self.real_to_complex(quantized_param)

            setattr(self, f"{param_name}", nn.Parameter(quantized_param))
            # self.register_parameter(f"q_{param_name}", nn.Parameter(quantized_param))
            self.register_buffer(f"{param_name}_scales", scales)
            self.register_buffer(f"{param_name}_zero_points", zero_points)

            if print_msg:
                print("Quantized weights for", param_name)

    def step(self, u, state, dA, dB, dC):
        Ax = torch.einsum("h n, b h n -> b h n", dA, state)
        Bu = torch.einsum("h n, b h -> b h n", dB, u)

         # Requantise state operations
        if self.state_quantizer is not None:
            Ax = self.real_to_complex(self.state_quantizer(self.complex_to_real(Ax)))
            Bu = self.real_to_complex(self.state_quantizer(self.complex_to_real(Bu)))
        
        next_state = Ax + Bu

        y = torch.einsum("c h n, b h n -> b c h", dC, next_state)
        return 2*y.real, next_state


    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (B H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        inputs = x

        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        # y, state = self.layer(x, **kwargs)
        # The following basically replaces the layer call ------------
            
        # Unparameterized A matrix
        if self.parameterized_A:
            dA = self.dA_inv_fn()
        else:
            dA = self.dA
            
        # Select quantised values ------------------------------------------------
        if not self.disable_quantization_flag:
            dA, dB, dC, D = self.weight_quantizer.get_quantized_weights(["dA", "dB", "dC", "D"], [self.complex_to_real(dA), self.complex_to_real(self.dB), self.complex_to_real(self.dC), self.D])
            dA, dB, dC = self.real_to_complex(dA), self.real_to_complex(dB), self.real_to_complex(dC)
        else:
            dA, dB, dC, D = getattr(self, "orig_dA", dA), getattr(self, "orig_dB", self.dB), getattr(self, "orig_dC", self.dC), getattr(self, "orig_D", self.D)
        # ----------------------------------------------------------------
        
        state = torch.zeros(x.size(0), self.d_model, self.N, device=x.device) # (B H N)
        all_states = torch.empty(x.size(0), self.d_model, self.N, L, device=x.device, dtype=torch.cfloat) # (B H L)

        y = torch.zeros(x.size(0), self.d_model, L, device=x.device) # (B H L)

        # print(dA)
        # print(dA.abs())
        # print(inputs.abs().max(), inputs.abs().min())
        # print("orig inputs", inputs)
        # print("inputs", x)

        # Recurrent loop
        for i in range(L):
            y_recur, state = self.step(x[:,i,:], state, dA, dB, dC) # (B, 1, H), (B H N)

            # Requantise state
            if self.state_quantizer is not None: # Have to call quantizer even when quantization disabled to collect statistics
                state = self.real_to_complex(self.state_quantizer(self.complex_to_real(state)))

            # Clamp values
            state = self.real_to_complex(torch.clamp(self.complex_to_real(state), min=self.state_clamp_min, max=self.state_clamp_max))
            y_recur = torch.clamp(y_recur, min=self.y_recur_clamp_min, max=self.y_recur_clamp_max)
            
            # TEST: Check for NaNs
            # if torch.isnan(state).any():
            #     print("NaNs in state", i)
            #     print(state)
            #     return
            # if y_recur.isnan().any():
            #     print("NaNs in y_recur", i)
            #     print(y_recur)
            #     return
            
            all_states[:, :, :, i] = state
            y[:, :, i] = y_recur.squeeze()

        # Requantise output
        if self.y_recur_quantizer is not None:
            y = self.y_recur_quantizer(y)

        # Add D
        y = y + torch.einsum("c h, b L h -> b h L", self.D, x)
        y = self.y_pre_output_quantizer(y) if self.y_pre_output_quantizer is not None else y
        
        pre_GELU_activation = y

        # Activation (GELU)
        y = self.layer_activation(y)

         # ^ Use this for statistics
        post_GELU_activation = y

        y = rearrange(y, 'b ... L -> b L ...')
        # ----------------------------------------------------------------

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y) # Linear + GLU

        # Should I requantise here?

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')
        
        # Output shape: (B H L)
        return y, (inputs, post_GELU_activation, pre_GELU_activation, all_states)
    