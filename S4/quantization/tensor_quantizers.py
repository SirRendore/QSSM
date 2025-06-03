import torch
from .q_utils import _tensor_quant, _tensor_dequant
from torch.autograd import Function



class TensorQuantFunction(Function):
    """A universal tensor quantization function

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    def forward(ctx, inputs, amax_min, mode="symmetric", zero_point_bits=16, num_bits=8, unsigned=False, narrow_range=False):
        """

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.
            scale: A Tensor of type float32. outputs / scale will dequantize outputs tensor.

        Raises:
            ValueError:
        """
        if num_bits < 2:
            raise ValueError("num_bits must be greater than 1")
        if isinstance(amax_min, tuple):
            amax, amin = amax_min
            # ctx.save_for_backward(inputs, amax, amin)
        else:
            amax = amax_min
            amin = None
        ctx.save_for_backward(inputs, amax, amin)

        outputs, scale, zero_point = _tensor_quant(inputs, amax_min, mode, zero_point_bits, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError("scale is too large for FP16 with amax_min={}".format(amax_min))
        
        return outputs, scale, zero_point
    
    @staticmethod
    def backward(ctx, grad_outputs, grad_scale):
        """
        Implements straight through estimation with clipping. For -amax <= input <= amax
        the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.
            grad_scale: A tensor of gradient of scale.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax, amin = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)  # create a zero tensor with the same type and device
        if amin is None:
            grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        else:
            grad_inputs = torch.where(inputs >= amin, grad_outputs, zero)
            grad_inputs = torch.where(inputs <= amax, grad_outputs, zero)

        return grad_inputs, None, None, None, None
    

class FakeTensorQuantFunction(Function):
    """A fake tensor quantization function

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    def forward(ctx, inputs, amax_min, mode="symmetric", zero_point_bits=16, num_bits=8, unsigned=False, narrow_range=False):
        """

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.

        Raises:
            ValueError:
        """
        if num_bits < 2:
            raise ValueError("num_bits must be greater than 1")
        if isinstance(amax_min, tuple):
            amax, amin = amax_min
        else:
            amax = amax_min
            amin = None
        ctx.save_for_backward(inputs, amax, amin)

        outputs, scale, zero_point = _tensor_quant(inputs, amax_min, mode, zero_point_bits, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError("scale is too large for FP16 with amax_min={}".format(amax_min))
        
        return _tensor_dequant(outputs, scale, zero_point, mode), scale, zero_point
    
    @staticmethod
    def backward(ctx, grad_outputs, grad_scale, grad_zero_point):
        """
        Implements straight through estimation with clipping. For -amax <= input <= amax
        the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax, amin = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)  # create a zero tensor with the same type and device
        if amin is None:
            grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        else:
            grad_inputs = torch.where(inputs >= amin, grad_outputs, zero)
            grad_inputs = torch.where(inputs <= amax, grad_outputs, zero)

        return grad_inputs, None, None, None, None, None, None

tensor_quant = TensorQuantFunction.apply
fake_tensor_quant = FakeTensorQuantFunction.apply

class TensorQuantizer(torch.nn.Module):
    '''
    Wrapper for tensor_quant and fake_tensor_quant
    '''

    def __init__(self, 
                 fake_real="real", 
                 percentile=100,
                 dims_to_reduce=None,
                 mode="symmetric", 
                 zero_point_bits=16, 
                 num_bits=8, 
                 unsigned=False, 
                 narrow_range=False):
        
        super().__init__()

        assert fake_real in ["fake", "real"], "fake_real must be either 'fake' or 'real'"
        assert mode in ["symmetric", "asymmetric"], "Unknown quantization mode: {mode}. Use symmetric or asymmetric."
        assert zero_point_bits > 0, "zero_point_bits must be greater than 0"
        assert num_bits > 0, "num_bits must be greater than 0"

        self.fake_real = fake_real
        self.percentile = percentile
        self.dims_to_reduce = dims_to_reduce
        self.mode = mode
        self.zero_point_bits = zero_point_bits
        self.num_bits = num_bits
        self.unsigned = unsigned
        self.narrow_range = narrow_range

    def get_amax_amin(self, inputs, custom_dims_to_reduce=None):
        '''
        Get amax and amin from inputs, based on percentile and dims_to_reduce
        '''
        input_device = inputs.device
        # inputs = inputs.cpu()

        if custom_dims_to_reduce is not None:
            dims_to_reduce = custom_dims_to_reduce
        else:
            dims_to_reduce = self.dims_to_reduce if self.dims_to_reduce is not None else range(inputs.ndim)

        if self.mode == "symmetric":
            if self.percentile == 100 or self.percentile is None:
                amax = torch.amax(inputs.abs(), dim=self.dims_to_reduce, keepdim=True)
            else:
                amax = inputs.to(torch.float32)
                for dim in dims_to_reduce:
                    amax = torch.quantile(amax.abs(), self.percentile/100, dim=dim, keepdim=True, interpolation='nearest')
            return amax.to(inputs.dtype).to(input_device)
            
        elif self.mode == "asymmetric":
            if self.percentile == 100 or self.percentile is None:
                amax = torch.amax(inputs, dim=self.dims_to_reduce, keepdim=True)
     
            else:
                amax = inputs.to(torch.float32)
                amin = inputs.to(torch.float32)
                for dim in dims_to_reduce:
                    amax = torch.quantile(amax, self.percentile/100, dim=dim, keepdim=True, interpolation='nearest')
                    amin = torch.quantile(amin, 1 - self.percentile/100, dim=dim, keepdim=True, interpolation='nearest')
            return amax.to(inputs.dtype).to(input_device), amin.to(inputs.dtype).to(input_device)

    def dequantize(self, inputs, scale, zero_point):
        '''
        Only for real quantization
        '''
        return _tensor_dequant(inputs, scale, zero_point, self.mode)

    def forward(self, inputs):
        amax_min = self.get_amax_amin(inputs)

        if self.fake_real == "fake":
            return fake_tensor_quant(inputs, amax_min, self.mode, self.zero_point_bits, self.num_bits, self.unsigned, self.narrow_range)
        elif self.fake_real == "real":
            return tensor_quant(inputs, amax_min, self.mode, self.zero_point_bits, self.num_bits, self.unsigned, self.narrow_range)
        
    def __repr__(self):
        return (
            f"{super(TensorQuantizer, self).__repr__()}\n"
            f"Quantization parameters: \n"
            f"  fake_real: {self.fake_real}\n"
            f"  percentile: {self.percentile}\n"
            f"  dims_to_reduce: {self.dims_to_reduce}\n"
            f"  mode: {self.mode}\n"
            f"  zero_point_bits: {self.zero_point_bits}\n"
            f"  num_bits: {self.num_bits}\n"
            f"  unsigned: {self.unsigned}\n"
            f"  narrow_range: {self.narrow_range}\n"
        )