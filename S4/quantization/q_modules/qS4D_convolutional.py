# S4D Conv Mode
from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
from .base_qmodule import BaseQModule
import torch.nn.functional as F


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
# Try CUDA extension
try:
    from extensions.kernels.cauchy import cauchy_mult as cauchy_cuda
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    print("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.")
except:
    print(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled."
    )
    has_cuda_extension = False

# Try pykeops
try:
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    print("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def cauchy_keops(v, z, w):
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde_keops(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return 2*_r2c(r).real

    def log_vandermonde_transpose_keops(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)
except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        print(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
        )
def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real

def log_vandermonde_transpose_naive(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
    return vandermonde_prod
# Function aliases
contract = torch.einsum

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


class S4BlockConvolutional(BaseQModule):
    """General quantisatble convolutional block for S4D model

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
        # self.s4_block = s4_block

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


       # Convolutional layer
        self.layer = s4_block.layer
        # ---------------------------

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = s4_block.mult_activation
        self.drop = s4_block.drop
        self.output_linear = s4_block.output_linear


        self.complex_to_real_mode = kwargs.get("complex_to_real_mode", "cartesian")
        self.complex_to_real, self.real_to_complex = c2r(self.complex_to_real_mode)

        # ---- Quantisation params
        self.weight_quantizer = None
        self.y_pre_output_quantizer = None
        self.register_buffer("disable_quantization_flag", torch.tensor(0)) # Register as buffer to store in state_dict


    def quantize_activations(self, act_quantizer):
        """
        Quantizes the activations of the block

        Args:
            act_quantizer (ActivationQuantizer): Activation quantizer object, a partial function
        """
        if act_quantizer is not None:
            # Custom quantizers for activations
            # if self.state_quantizer_bits is not None:
            #     self.state_quantizer = act_quantizer(num_bits=self.state_quantizer_bits, dims_to_reduce = self.state_quantizer_dims)
            # else:
            #     self.state_quantizer = act_quantizer(dims_to_reduce = self.state_quantizer_dims) # (B H N C=2) per channel            
            # self.y_recur_quantizer = act_quantizer(dims_to_reduce = [-2]) # # (B H L) per channel
            self.y_pre_output_quantizer = act_quantizer(dims_to_reduce=[-2]) # (B H L) 
            print("Quantized activations inside S4BlockConvolutional")
            
        #     # Register hook on backward of state quantizer if clipping set
        #     if self.state_quantizer_clip is not None:
        #         self.state_quantizer_handle = self.state_quantizer.register_full_backward_hook(lambda module, grad_input, grad_output: (torch.clamp(grad_input[0], min=-self.state_quantizer_clip, max=self.state_quantizer_clip), ))
        #         print("Registered clipping hook on state quantizer")
        # else:
        #     print("No activation quantizer provided for S4BlockConvolutional")
        pass
        

    def disable_quantization(self):
        """
        Disables quantization for the block

        """
        self.disable_quantization_flag = torch.tensor(1)
        # self.state_quantizer.disable_quantization()
        # self.y_recur_quantizer.disable_quantization()
        # self.y_pre_output_quantizer.disable_quantization()
        print(f"Disabled quantization for S4BlockConvolutional")

    def enable_quantization(self):
        '''
        Enables quantization for the block
        '''
        self.disable_quantization_flag = torch.tensor(0)
        # self.state_quantizer.enable_quantization()
        # self.y_recur_quantizer.enable_quantization()
        # self.y_pre_output_quantizer.enable_quantization()
        print("Enabled quantization for S4BlockConvolutional")
        

    def quantize(self, weight_quantizer, print_msg=True):
        """
        Quantizes the weights of the block

        Args:
            weight_quantizer (ModuleWeightQuantizer): Weight quantizer object
            print_msg (bool): Print message if True
        """
       
        if self.weight_quantizer is None:
            self.weight_quantizer = weight_quantizer

        print("Stored weight quantizer in S4BlockConvolutional")
        

    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

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

        # --------- FFT Conv ------------
        # y, state = self.layer(x, **kwargs)
        # state = torch.zeros(x.size(0), self.d_model, self.layer.kernel.N, device=x.device)
        state = kwargs.get('state', None)
            
        # Always work with (B D L) dimension in this module
        if not self.layer.transposed: layer_x = x.transpose(-1, -2)
        layer_L = layer_x.size(-1)

        # Compute SS Kernel
        l_kernel = layer_L if self.layer.L is None else min(layer_L, round(self.layer.L))

        # ---------- Kernel ----------
        # k, k_state = self.layer.kernel(L=l_kernel) # (C H L) (B C H L)
        L = l_kernel
        dt, A, B, C = self.layer.kernel._get_params(1)
        dtA = dt * A

        # Discretise
        if self.layer.kernel.disc == 'zoh':
            dA = torch.exp(dtA) # (H N)
            dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.layer.kernel.disc == 'bilinear':
            dA = (1. + dtA/2) / (1. - dtA/2)
            dB = B * (1. - dtA/2).reciprocal() * dt # or * dtA / A
        else: raise ValueError(f"Discretization {self.layer.kernel.disc} not supported")
        dB = rearrange(dB, '1 h n -> h n')
        dC = C
        dD = self.layer.D

        if not self.disable_quantization_flag:
            dA, dB, dC, D = self.weight_quantizer.get_quantized_weights(["dA", "dB", "dC", "D"], [self.complex_to_real(dA), self.complex_to_real(dB), self.complex_to_real(dC), dD])
            dA, dB, dC = self.real_to_complex(dA), self.real_to_complex(dB), self.real_to_complex(dC)

        # Augment B with state
        if state is not None:
            s = state / dt
            if self.layer.kernel.disc == 'bilinear':
                s = s * (1. + dtA/2)
            elif self.layer.kernel.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3) # (1+B H N)

        # Combine B and C
        C = (dB * dC).view(-1, self.layer.kernel.H, self.layer.kernel.N)

        # Dispatch which Vandermonde kernel to use
        if has_cuda_extension and C.dtype == torch.cfloat and C.device.type == 'cuda' and self.layer.kernel.backend == 'cuda':
            log_vandermonde = log_vandermonde_cuda
        elif has_pykeops and self.layer.kernel.backend in ['cuda', 'keops']:
            log_vandermonde = log_vandermonde_keops
        else:
            log_vandermonde = log_vandermonde_naive

        # Main kernel
        K = log_vandermonde(C, dA.log(), L)

        K = K.view(-1, self.layer.kernel.channels, self.layer.kernel.H, L) # (1+B C H L)

        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)

        # # -----------------------------
        k, k_state = K, K_state

        # Convolution
        if self.layer.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, layer_L)) \
                    + F.pad(k1.flip(-1), (layer_L, 0))

        # Kernel dropout
        k = self.layer.drop_kernel(k)

        # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
        # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
        k_f = torch.fft.rfft(k, n=l_kernel+layer_L) # (C H L)
        x_f = torch.fft.rfft(layer_x, n=l_kernel+layer_L) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel+layer_L)[..., :layer_L] # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', layer_x, dD)

        # Compute state update
        if state is not None:
            assert not self.layer.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.layer.kernel.forward_state(layer_x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.layer.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.layer.drop(y)  # DropoutNd better with transposed=True

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.y_pre_output_quantizer(y) if self.y_pre_output_quantizer is not None else y
        pre_GELU_activation = y
        y = self.layer.activation(y)
        post_GELU_activation = y

        
        y = rearrange(y, 'b ... L -> b L ...')
        # ----------------------------------
        y, state = y, next_state


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, (inputs, post_GELU_activation, pre_GELU_activation, None)