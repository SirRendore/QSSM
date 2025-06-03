import torch.nn as nn
import numpy as np
from train import set_up_trainer
import argparse
from parse_arguments import ConfigParser

def main(config):
    '''
    Set up trainer object, then call eval method
    '''
    trainer = set_up_trainer(config)
    trainer.eval()

    return trainer

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, save_log=False) # Don't save eval runs
    main(config)