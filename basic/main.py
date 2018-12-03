#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:35:00 2018

@author: xavier.qiu
"""
import os
import shutil
import tensorflow as tf
import argparse
import json

def main(config):
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)
        else:
            _forward(config)
    pass


def set_dirs(config):
    
    assert config.load or config.mode == 'train'
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)
        
    config.save_dir = os.path.join(config.out_dir, "save")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    
    makedir_if_not_exist(config.out_dir)
    makedir_if_not_exist(config.save_dir)
    makedir_if_not_exist(config.eval_dir)
    makedir_if_not_exist(config.answer_dir)
    makedir_if_not_exist(config.log_dir)
    
def makedir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _train(config):
    pass

def _test(config ): 
    pass

def _forward(config ):
    pass

def _get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()

def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)

if __name__ == "__main__":
    _run()
    
    #%%
    
#%%