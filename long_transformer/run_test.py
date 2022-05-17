#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from re import L

from numpy import load
from models.model_utils import build_model
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
    
    run_config.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    run_config.world_size = len(run_config.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = run_config.visible_gpus

    init_logger(run_config.log_file)
    device = "cpu" if run_config.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (run_config.task == 'abs'):
        if (run_config.mode == 'train'):
            train_abs(run_config, device_id)
        elif (run_config.mode == 'validate'):
            validate_abs(run_config, device_id)
        elif (run_config.mode == 'lead'):
            baseline(run_config, cal_lead=True)
        elif (run_config.mode == 'oracle'):
            baseline(run_config, cal_oracle=True)
        if (run_config.mode == 'test'):
            cp = run_config.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(run_config, device_id, cp, step)
        elif (run_config.mode == 'test_text'):
            cp = run_config.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(run_config, device_id, cp, step)

    elif (run_config.task == 'ext'):
        if (run_config.mode == 'train'):
            train_ext(run_config, device_id)
        elif (run_config.mode == 'validate'):
            validate_ext(run_config, device_id)
        if (run_config.mode == 'test'):
            cp = run_config.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(run_config, device_id, cp, step)
        elif (run_config.mode == 'test_text'):
            cp = run_config.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(run_config, device_id, cp, step)
