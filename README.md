# QSSM
A quantization pipeline for SSMs


The framework is built on top of PyTorch, and is designed to be modular and flexible. 

# Basics

The codebase enables initialisation, training and evaluation of both full-precision and quantised models, as can be defined by the configuration files. The configuration files are written in YAML format and define the model architecture and hyperparameters. An additional configuration file defines the quantisation scheme and precision.

The full-precision model is defined in the `models` directory. Intialisation, training and evaluation configuration are specified in the configuration file in the `configs` directory. After training, the model can be quantised using the logic in `quantize.py`. The quantised model can be trained with QAT, or immediately evaluated with just PTQ.

This framework permits static and dynamic quantisation of weights and activations, with support for arbitrary bit precisions, symmetric and asymmetric, per-channel and per-tensor quantisation schemes, as well as different calibration modes. As can be specified in the quantisation configuration file, each weight and activation can be quantised independently, using different schemes.

Quantisation logic is contained within the `quantization` directory. The `q_modules` directory contains the quantisable modules, which are custom, user-defined PyTorch modules that can be quantised. These modules replace specified modules in the full-precision model and inherit from an abstract class defined in `base_qmodules.py`, which defines the interface for quantisation. 

A distinction is made between weight and activation quantisation. Weight quantisation is straightforward as the weights are fixed during inference and can simply be quantised once and offline. The quantisers for weights are defined in `tensor_quantizers.py`. Quantisation of activations are performed using activation quantisers, also defined in `tensor_quantizers.py`. Quantiser modules are inserted where necessary to quantise activations during the forward pass. These quantisers can perform dynamic calibration and contain the necessary observers for static calibration, as well as the logic for the straight-through gradient estimator for the backward pass of QAT. 

In general, when changing the quantisation scheme, the user need only modify the quantisation configuration file. Training and evaluation are compatible with both the full-precision and quantised models, allowing for easy comparisons between the full-precision model and the quantised model with different quantisation schemes.

# Usage examples

## Specifying quantisation config

For each weight or activation, the following quantisation parameters can be
specified in the quantisation configuration file:

- **num_bits**: Bit precision of the quantised values.
• mode: Quantisation mode, either symmetric or asymmetric.
- **dims_to_reduce**: Defines the granularity of quantisation. The dimensions
to reduce can be specified as a list of integers. Reducing over all dimensions results in per-tensor quantisation, while reducing over a single dimension
results in per-channel quantisation.
- **percentile**: Percentile value used for percentile calibration. If set to 100,
max-min calibration is used.
- **zero_point_bits**: Bit precision of the zero-point. Only used in asymmetric
quantisation.
- **unsigned**: Whether the quantised values are unsigned. If set to True, the
quantised range is [0, 2^(num_bits − 1)]. If set to False, negative values are
included in the quantised range. False by default.
- **narrow_range**: Whether the narrow-range quantised range is used for asymmetric
quantisation. False by default. Note that symmetric quantisation always uses narrow range.

## Adding a quantisable module

Quantisable modules, which we will call `QModules`, are custom PyTorch modules
that can be quantised. To add a quantisable module, the user must define a
new module that inherits from the abstract class `BaseQModule` defined in `base_qmodules.py`. The user must define the forward pass of the module, as well as the quantisation logic.

The `BaseQModule` class defines four functions:

- **quantize**: Quantises the weights of the module given a weight quantiser
which defines the quantisation scheme for each weight.
- **quantize_activations**: Quantises the activations of the module given an
activation quantiser which defines the quantisation scheme for the activations.
This is used if the module contains intermediate activations that must be
quantised, such as the state in the SSM.
- **disable_quantization**: Sets the module to full-precision mode. This is
used during calibration.
- **enable_quantization**: Sets the module to quantised mode.

The `QModule` must also be initialised with the full-precision module it replaces.
In other words, its initialisation function should accept an instance of the fullprecision
module, from which the weights and other properties can be copied.

Once the `QModule` is defined, the `qModules` class in the `__init__.py` file in
q_modules must be updated to include the new `QModule` as well as the selection
of full-precision modules it is permitted to replace. The full-precision modules
can be replaced with the `QModule` by setting the `target` field in the quantisation
configuration file. The `QModule` is then be quantised according to the specified
quantisation scheme.


## Static, dynamic, QAT

For *PTQ*, the user simply specifies the quantisation scheme in the quantisation
configuration file, then calls `set_up_quantizers` from the `quantize.py` script on
a pre-trained model. By default, if no calibration is done, dynamic quantisation
is performed.

For *static* quantisation, the user must first *calibrate* the model on a calibration
dataset. This is done by calling the `calibrate` function attached to the `Trainer`
class in the `train.py` script. The user can specify the number of batches used
to construct the calibration dataset out of the training dataset. The calibration
dataset is then fed through the model while the activation quantisers collect
statistics to determine the appropriate quantisation ranges. Once calibration
is complete, the model can be configured to use static quantisation by calling
`set_static_quantization` attached to the `Trainer` class. To revert to dynamic
quantisation, the user can simply call `set_dynamic_quantization`. These functions
also accept an argument to specify the modules to be excluded from being
set to static or dynamic quantisation, enabling the possibility for some modules
to be quantised dynamically while others are quantised statically.
At any point, the user can call `disable_all_quantization` and `enable_all_quantization`, both attached to the `Trainer` class, to disable or enable
quantisation for all modules in the model. Specific modules can also be excluded
from this operation by specifying the modules to exclude as an argument. This allows for easy switching between full-precision and quantised models during
training and evaluation.

For *QAT*, the user simply calls the `train` function attached to the `Trainer`
class, once the quantisers have been set up. The user can specify the learning
rate, number of epochs and other training hyperparameters in the quantisation
configuration file.

## Examples

See `/S4/scripts/examples.ipynb`.