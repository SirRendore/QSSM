from S4.models.S4 import S4Block   
import torch
from functools import partial


def modify_blocks(module, to_replace, replaced_class, module_name_to_exclude = [], quantize_block=False, weight_quantizer=None, activation_quantizer=None, replaced_class_args={}):
    for name, child in module.named_children():
        if isinstance(child, to_replace) and not any([x == name for x in module_name_to_exclude]):
            new_module = replaced_class(child, **replaced_class_args)                                        
            
            if quantize_block:
                print(f"[*] Quantizing {name}")
                assert weight_quantizer is not None, "Weight quantizer must be provided to quantize block"
                new_module.quantize(weight_quantizer)

            print(f"[*] Replacing {name} with {new_module}")
            setattr(module, name, new_module)
                

        else:
            # Recursively call the function for nested modules
            modify_blocks(child, to_replace, replaced_class, module_name_to_exclude, quantize_block=quantize_block, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer, replaced_class_args=replaced_class_args)


def insert_blocks(module, to_insert, target_class, module_name_to_exclude = []):
    """
    Instantiates and inserts a block before a target class in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, target_class) and not any([x == name for x in module_name_to_exclude]):
            to_insert_inst = to_insert()
            print(f"[*] Inserting {to_insert_inst} before {name}")
            try:
                child.quantize_activations(to_insert)
                print(f"[*] Passed in {to_insert} to {name}")
            except AttributeError:
                print(f"[*] {name} does not have a quantize_activations method. Inserting {to_insert_inst} before {name}")

            setattr(module, name, torch.nn.Sequential(to_insert_inst, child))
        else:
            # Recursively call the function for nested modules
            insert_blocks(child, to_insert, target_class, module_name_to_exclude)

def apply_function_to_module(module, function, module_name_to_exclude = []):
    '''
    Call function on all modules and nested modules
    '''
    for name, child in module.named_children():
        if not any([x == name for x in module_name_to_exclude]):
            function(child)
        # Recursively call the function for nested modules
        apply_function_to_module(child, function, module_name_to_exclude)

def _tensor_quant(inputs, amax_min, mode="symmetric", zero_point_bits=16, num_bits=8, unsigned=False, narrow_range=False):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
    
    # Check inputs
    if isinstance(amax_min, tuple):
        amax, amin = amax_min
    else:
        amax = amax_min
        amin = None

    assert mode in ["symmetric", "asymmetric"], f"Unknown quantization mode: {mode}. Use symmetric or asymmetric."
    if mode == "asymmetric":
        assert amin is not None, "Asymmetric quantization requires both amax and amin."
        assert amin.shape == amax.shape, f"amin ({amin.shape}) and amax ({amax.shape}) must have the same shape."

    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
        print("amax %s has different shape than inputs %s. Make sure broadcast works as expected!", amax.size(),
                      inputs.size())

    # print("{} bits quantization on shape {} tensor.".format(num_bits, inputs.size()))

    if unsigned:
        if inputs.min() < 0.:
            raise TypeError("Negative values encountered in unsigned quantization.")
        

    # Computation must be in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    if mode == 'symmetric' and min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    z_max_bound = torch.tensor(2.0**(zero_point_bits - 1) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1

    if mode == "symmetric":
        scale = amax / max_bound 

        epsilon = 1. / (1 << 24)
        if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
            zero_amax_mask = (amax <= epsilon)
            scale[zero_amax_mask] = 1  # Value quantized with amax=0 should all be 1
        outputs = torch.clamp((inputs / scale).round_(), min_bound, max_bound)
        
        if input_dtype == torch.half:
            outputs = outputs.half()
        
        return outputs, scale, None # zero_point is None for symmetric quantization

    elif mode == "asymmetric":
        scale = (amax - amin) / (max_bound - min_bound)
        
        epsilon = 1. / (1 << 24)
        if (amax-amin).min() <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
            zero_amax_mask = ((amax-amin) <= epsilon)
            scale[zero_amax_mask] = 1  # Value quantized with amax=0 should all be 0
       
        zero_point = min_bound - amin / scale
        zero_point = torch.clamp(zero_point.round_(), -z_max_bound - 1, z_max_bound)
        outputs = torch.clamp((inputs / scale + zero_point).round_(), min_bound, max_bound)
       
        if input_dtype == torch.half:
            outputs = outputs.half()

        return outputs, scale, zero_point
    
def _tensor_dequant(inputs, scale, zero_point, mode="symmetric"):
    """Shared function body between TensorDequantFunction and FakeTensorDequantFunction"""

    # Check inputs
    assert mode in ["symmetric", "asymmetric"], f"Unknown quantization mode: {mode}. Use symmetric or asymmetric."

    if mode == "asymmetric":
        assert zero_point is not None, "Asymmetric dequantization requires zero_point."

    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if isinstance(scale, torch.Tensor) and inputs.dim() != scale.dim():
        print("scale %s has different shape than inputs %s. Make sure broadcast works as expected!", scale.size(),
                      inputs.size())

    # print("Dequantization on shape {} tensor.".format(inputs.size()))

    if mode == "symmetric":
        outputs = inputs * scale
        return outputs.to(scale.dtype)

    elif mode == "asymmetric":
        outputs = (inputs - zero_point) * scale
        return outputs.to(scale.dtype)

