import os
import torch
import numpy as np
from tqdm import tqdm
from utils import ensure_dir, prepare_device
from parse_arguments import ConfigParser
from S4.dataloaders import Datasets
from S4.models import Models, Losses
import argparse
import collections
import torchsummary
from quantize import (prepare_for_calibration, finish_calibration, 
                      disable_all_quantization, enable_all_quantization, 
                      set_static_quantization, set_dynamic_quantization, 
                      quantize_all_weights, update_q_trainer_for_QAT)
from functools import reduce

class Trainer():
    def __init__(self, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, 
                 trainloader: torch.utils.data.DataLoader, 
                 device: torch.device, 
                 config: ConfigParser, 
                 writer=None,
                 valloader=None, 
                 testloader=None, 
                 lr_scheduler=None, 
                 epoch=1):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_clip = config["train"]["grad_clip"] if "grad_clip" in config["train"] else None

        self.epochs = config["train"]["epochs"]
        self.save_period = config["train"]["save_period"] if "save_period" in config["train"] else 1

        self.config = config
        self.device = device
        self.trainloader = trainloader

        self.valloader = valloader
        self.do_validation = self.valloader is not None
        self.testloader = testloader
        self.do_testing = self.testloader is not None

        self.do_save_checkpoint = config["train"]["save_checkpoint"]

        self.lr_scheduler = lr_scheduler

        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_test_acc = 0

        self.writer = writer

        if config.save_dir is None:
            pwd = os.getcwd()
            save_dir = os.path.join( pwd, 'S4/log' )
            os.makedirs( save_dir )
            self.save_dir = save_dir
        else:
            self.save_dir = config.save_dir

        self.start_epoch = epoch
        if config.resume:
            self.start_epoch = self.load_checkpoint(config.resume)
        self.current_epoch = self.start_epoch


    def save_checkpoint(self, epoch):
        if not self.do_save_checkpoint:
            return

        print(f"Best model found. Saving checkpoint at {self.save_dir}")
        state = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            "best_train_acc": self.best_train_acc   
        }
        if self.lr_scheduler is not None:
            state['scheduler'] = self.lr_scheduler.state_dict()
        if self.do_validation:
            state['best_val_acc'] = self.best_val_acc
        if self.do_testing:
            state['best_test_acc'] = self.best_test_acc

        ensure_dir(self.save_dir / 'checkpoint')
        torch.save(state, self.save_dir / 'checkpoint' / 'ckpt.pth')


    def load_checkpoint(self, checkpoint):
        print("Loading checkpoint from: ", checkpoint)

        checkpoint = torch.load(checkpoint)

        # Update buffer size of amax/amin in all activation quantizers, otherwise get size mismatch error
        # Hacky... don't love this
        def get_from_model(model, dict_list):
            return reduce(getattr, dict_list, model)
        for k, v in checkpoint["model"].items():
            key_list = k.split(".")
            if key_list[-1] in ["amax", "amin"] or "scales" in key_list[-1] or "zero_points" in key_list[-1]:
                setattr(get_from_model(self.model, key_list[:-1]), key_list[-1], torch.zeros_like(v))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        self.best_train_acc = checkpoint['best_train_acc']
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
        if 'best_test_acc' in checkpoint:
            self.best_test_acc = checkpoint['best_test_acc']
        if 'scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        return epoch
    
    def freeze_modules(self, module_name_to_exclude=[]):
        """
        Freeze all parameters in the module except for those in the modules specified in module_name_to_exclude
        """

        def _freeze_modules(module):
            for name, child in module.named_children():
                if not any([isinstance(child, x) for x in module_name_to_exclude]):
                    print(f"Freezing parameters in {name}")
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    print(f"Unfreezing {name}")
                    for param in child.parameters():
                        param.requires_grad = True
                # Recursively call the function for nested modules
                _freeze_modules(child)

        _freeze_modules(self.model)

    def unfreeze_modules(self, module_name_to_exclude=[]):
        """
        Unfreeze all parameters in the module except for those in the modules specified in module_name_to_exclude
        """
        def _unfreeze_modules(module):
            for name, child in module.named_children():
                if not any([isinstance(child, x) for x in module_name_to_exclude]):
                    print(f"Unfreezing parameters in {name}")
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    print(f"Freezing {name}")
                    for param in child.parameters():
                        param.requires_grad = False
                # Recursively call the function for nested modules
                _unfreeze_modules(child)

        _unfreeze_modules(self.model)

    def train_epoch(self, epoch):
        log = {}

        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(enumerate(self.trainloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device) # Squeeze the inputs
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'TRAIN - Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(self.trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss/(batch_idx + 1), epoch)
            self.writer.add_scalar("Accuracy/train", 100.*correct/total, epoch)

        if 100.*correct/total > self.best_train_acc:
            self.best_train_acc = 100.*correct/total
            if not self.do_validation and epoch % self.save_period == 0:
                self.save_checkpoint(epoch)

        if self.do_validation:
            val_log = self.validate_epoch(epoch) 
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.do_testing:
            test_log = self.test_epoch(epoch, writer=self.writer)
            log.update(**{'test_'+k : v for k, v in test_log.items()})

        if self.lr_scheduler:
            self.lr_scheduler.step()

        log["epoch"] = epoch
        log["train_loss"] = train_loss/(batch_idx + 1)
        log["train_acc"] = 100.*correct/total
        log["lr"] = self.lr_scheduler.get_last_lr() if self.lr_scheduler else None
        
        return log


    def validate_epoch(self, epoch):
        log = {}

        if self.valloader is None:
            raise ValueError("Validation loader not provided.")
        
        pbar = tqdm(enumerate(self.valloader))
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_description(
                    'VAL - Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(self.valloader), val_loss/(batch_idx+1), 100.*correct/total, correct, total)
                )

        if self.writer:
            self.writer.add_scalar("Loss/val", val_loss/(batch_idx + 1), epoch)
            self.writer.add_scalar("Accuracy/val", 100.*correct/total, epoch)

        if 100.*correct/total > self.best_val_acc:
            self.best_val_acc = 100.*correct/total
            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch)

        log["loss"] = val_loss/(batch_idx + 1)
        log["acc"] = 100.*correct/total
        
        return log

    def test_epoch(self, epoch, writer=None):
        log = {}

        if self.testloader is None:
            raise ValueError("Testloader not provided.")

        pbar = tqdm(enumerate(self.testloader))
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_description(
                    'TEST - Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(self.testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total)
                )

        if writer:
            writer.add_scalar("Loss/test", test_loss/(batch_idx + 1), epoch)
            writer.add_scalar("Accuracy/test", 100.*correct/total, epoch)

        if 100.*correct/total > self.best_test_acc:
            self.best_test_acc = 100.*correct/total

        log["loss"] = test_loss/(batch_idx + 1)
        log["acc"] = 100.*correct/total

        return log
    
    def disable_all_quantization(self, module_name_to_exclude=[]):
        '''Disable all quantization in the model'''
        disable_all_quantization(self.model, module_name_to_exclude=module_name_to_exclude)

    def enable_all_quatization(self, module_name_to_exclude=[]):
        '''Enable all quantization in the model'''
        enable_all_quantization(self.model, module_name_to_exclude=module_name_to_exclude)

    def set_dynamic_quantization(self, module_name_to_exclude=[]):
        '''Set dynamic quantization for all quantizers in the model'''
        set_dynamic_quantization(self.model, module_name_to_exclude=module_name_to_exclude)

    def set_static_quantization(self, module_name_to_exclude=[]):
        '''Set static quantization for all quantizers in the model'''
        set_static_quantization(self.model, module_name_to_exclude=module_name_to_exclude)

    def quantize_all_weights(self, module_name_to_exclude=[]):
        '''Quantize all weights in the model'''
        quantize_all_weights(self.model, module_name_to_exclude=module_name_to_exclude)

    def update_q_trainer_for_QAT(self):
        '''Update the quantizer trainer for quantization aware training'''
        update_q_trainer_for_QAT(self)

    def calibrate(self, num_batches = 1):
        '''
        Calibrate the model for quantization using the first num_batches batches of the training set
        '''
        
        print(f"Starting calibration with {num_batches} batches")
        self.disable_all_quantization()
        prepare_for_calibration(self.model)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if batch_idx >= num_batches:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

        finish_calibration(self.model)
        self.enable_all_quatization()
        print("Finished calibration")

    def train(self): 
        '''Main training loop'''

        for epoch in range(self.start_epoch, self.epochs + 1):
            result_log = self.train_epoch(epoch)       
            print(f'Epoch {epoch}/{self.epochs}, Train Acc: {result_log["train_acc"]}, Val Acc: {result_log["val_acc"]}, Test Acc: {result_log["test_acc"]}, LR: {self.lr_scheduler.get_last_lr()}')                  
            self.current_epoch += 1
        return result_log
        

    def eval(self):
        result_log = self.test_epoch(self.current_epoch)
        print(f'Eval Acc: {result_log["acc"]}, Loss: {result_log["loss"]}')
        return result_log


def set_up_trainer(config):
    '''
    Set up the trainer object. Set up the model, optimizer, dataloaders, loss function, learning rate scheduler, and writer.
    '''
    # set seeds
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # set up dataloaders
    train_loader, val_loader, test_loader, constants = config.init_obj('dataloaders', Datasets)

    # set up model
    model = config.init_obj('model', Models)
    model.name = config['model']['type']
    torchsummary.summary(model, train_loader.dataset[0][0].shape)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # set up optimizer, learning rate scheduler, loss function
    criterion = config.init_obj('loss', Losses)
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # set up writer
    writer = None
    if config['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=config.save_dir)

    trainer = Trainer(model, criterion, optimizer, train_loader, device, config,
                        writer=writer,
                        valloader=val_loader, 
                        testloader=test_loader, 
                        lr_scheduler=lr_scheduler)

    return trainer


def main(config):
    trainer = set_up_trainer(config)
    
    trainer.train()  
    if trainer.writer:
        trainer.writer.flush()
        trainer.writer.close()

    return trainer


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloaders;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)