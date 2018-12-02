#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:35:00 2018

@author: xavier.qiu
"""

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
    pass

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