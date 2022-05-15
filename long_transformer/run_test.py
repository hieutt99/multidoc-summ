#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os

from numpy import load
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext
from utils.arguments import load_config
from utils.arguments import RunConfig, ModelConfig

from dataloader import dataloader 
from dataloader.dataloader import load_dataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", default='run_config.yaml', type=str)

    args = parser.parse_args()

    run_config = load_config(args.run_config)

    device = "cpu" if run_config.visible_gpus == '-1' else "cuda"
    dataloader.Dataloader(run_config, load_dataset(run_config, 'train', shuffle=True), run_config.batch_size, device,
                                      shuffle=True, is_test=False)

    init_logger(run_config.log_file)
    pass 
