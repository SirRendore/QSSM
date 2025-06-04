import torch
import numpy as np
from S4.quantization.quantizers import ModuleWeightQuantizer, ActivationQuantizer
from S4.quantization.q_modules import qModules
from S4.quantization.q_utils import modify_blocks, insert_blocks, apply_function_to_module
import functools
from S4.models import Losses

def prepare_for_calibration(module, module_name_to_exclude=[]):
    # Set up statistics collection for activation quantizers
    for name, child in module.named_children():
        if isinstance(child, ActivationQuantizer) and not any([x == name for x in module_name_to_exclude]):
            child.setup_statistics_collection()
            print(f"Starting statistics collection for {name}")
        else:
            # Recursively call the function for nested modules
            prepare_for_calibration(child, module_name_to_exclude)

def finish_calibration(module, module_name_to_exclude=[]):
    # Finish statistics collection for activation quantizers
    for name, child in module.named_children():
        if isinstance(child, ActivationQuantizer) and not any([x == name for x in module_name_to_exclude]):
            print(f"Ending statistics collection for {name}")
            child.finish_statistics_collection()
        else:
            # Recursively call the function for nested modules
            finish_calibration(child, module_name_to_exclude)

def disable_all_quantization(module, module_name_to_exclude=[]):
    for name, child in module.named_children():
        if hasattr(child, "disable_quantization") and not any([x == name for x in module_name_to_exclude]):
            print(f"Disabling quantization for {name}")
            child.disable_quantization()
        
        # Recursively call the function for nested modules
        disable_all_quantization(child, module_name_to_exclude)

def enable_all_quantization(module, module_name_to_exclude=[]):
    for name, child in module.named_children():
        if hasattr(child, "enable_quantization") and not any([x == name for x in module_name_to_exclude]):
            child.enable_quantization()
        
        # Recursively call the function for nested modules
        enable_all_quantization(child, module_name_to_exclude)

def set_static_quantization(module, module_name_to_exclude=[]):
    def to_call_on_child(child):
        if hasattr(child, "set_static_quantization"):
            child.set_static_quantization()
        
    apply_function_to_module(module, to_call_on_child, module_name_to_exclude)

def set_dynamic_quantization(module, module_name_to_exclude=[]):
    def to_call_on_child(child):
        if hasattr(child, "set_dynamic_quantization"):
            child.set_dynamic_quantization()
        
    apply_function_to_module(module, to_call_on_child, module_name_to_exclude)
        
def quantize_all_weights(module, module_name_to_exclude=[]):
    '''
    Call quantize on all modules that have a quantize method 
    '''
    def to_call_on_child(child):
        if hasattr(child, "quantize") and hasattr(child, "weight_quantizer"):
            child.quantize(child.weight_quantizer)

    apply_function_to_module(module, to_call_on_child, module_name_to_exclude)

def set_up_quantizers(trainer, config):
    # set seeds
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    model = trainer.model
    model_q_args = config["model"][model.name]

    for qBlock in model_q_args:
        if "weights" in model_q_args[qBlock]: 
            weight_args = model_q_args[qBlock]["weights"]
            for weight in weight_args:
                weight_args[weight]["fake_real"] = model_q_args[qBlock]["fake_real"]
            weight_quantizer = ModuleWeightQuantizer(weight_args)
        else :
            weight_quantizer = ModuleWeightQuantizer({}) # Dummy weight quantizer

        if "target" in model_q_args[qBlock]:  # Replace modules if necessary
            # print(f"Replacing {model_q_args[qBlock]["target"]} with {qBlock}")
            qBlock_class = getattr(qModules, qBlock)["class"]
            try:
                target_class = getattr(qModules, qBlock)["allowed_targets"][model_q_args[qBlock]["target"]]
            except KeyError:
                raise KeyError(f"Target {model_q_args[qBlock]['target']} not allowed for {qBlock}")

            qBlock_class_args = model_q_args[qBlock]["args"] if "args" in model_q_args[qBlock] else {}
            modify_blocks(model, target_class, qBlock_class, quantize_block=True, weight_quantizer=weight_quantizer, replaced_class_args=qBlock_class_args)
        elif "weights" in model_q_args[qBlock]: # Quantize weights directly if no replacement
            print(f"Quantizing {qBlock}")
            getattr(model, qBlock).quantize(weight_quantizer)

        if "activations" in model_q_args[qBlock]:
            act_args = model_q_args[qBlock]["activations"]
            act_args["fake_real"] = model_q_args[qBlock]["fake_real"]

            try:
                qBlock_class = getattr(qModules, qBlock)["class"]
            except AttributeError:
                print(f"Could not find {qBlock} in qModules. Using torch.nn.{qBlock}")
                qBlock_class = getattr(torch.nn, qBlock)

            # Partially instantiate ActivationQuantizer to pass into insert_blocks, overwrite inside replaced module
            partial_module_aq = functools.partial(ActivationQuantizer, **act_args)

            insert_blocks(model, partial_module_aq, qBlock_class)
            print(f"Quantized input activations of {qBlock}. Inserted {partial_module_aq} before {qBlock}")

    # Update trainer values - kind of ugly here
    trainer.q_config = config

    if "loss" in config.config:
        trainer.criterion = config.init_obj('loss', Losses)
    if "train" in config.config:
        trainer.grad_clip = config["train"]["grad_clip"] if "grad_clip" in config["train"] else None
        trainer.epochs = config["train"]["epochs"]
        trainer.save_period = config["train"]["save_period"] if "save_period" in config["train"] else 1
        trainer.do_save_checkpoint = config["train"]["save_checkpoint"] if "save_checkpoint" in config["train"] else False
    
    if "optimizer" in config.config:
        trainer.optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    elif hasattr(trainer, "optimizer"):
        trainer.optimizer = trainer.config.init_obj('optimizer', torch.optim, model.parameters())
    
    if "lr_scheduler" in config.config:
        trainer.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, trainer.optimizer)
    elif hasattr(trainer, "lr_scheduler"):
        trainer.lr_scheduler = trainer.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, trainer.optimizer)

    return trainer

def update_q_trainer_for_QAT(trainer):
    # Update trainer values 
    try:
        config = trainer.q_config
    except AttributeError:
        raise AttributeError("Trainer does not have q_config attribute. Please set up quantizers first.")

    if "train" in config.config:
        trainer.epochs = config["train"]["epochs"]
        trainer.save_period = config["train"]["save_period"] if "save_period" in config["train"] else 1
        trainer.do_save_checkpoint = config["train"]["save_checkpoint"] if "save_checkpoint" in config["train"] else False

    if "optimizer" in config.config:
        trainer.optimizer = config.init_obj('optimizer', torch.optim, trainer.model.parameters())
    elif hasattr(trainer, "optimizer"):
        trainer.optimizer = trainer.config.init_obj('optimizer', torch.optim, trainer.model.parameters())
    
    if "lr_scheduler" in config.config:
        trainer.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, trainer.optimizer)
    elif hasattr(trainer, "lr_scheduler"):
        trainer.lr_scheduler = trainer.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, trainer.optimizer)


# TODO: Commmand line interface
