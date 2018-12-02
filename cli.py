#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:34:05 2018

@author: xavier.qiu
"""

import os
import tensorflow as tf

from basic.main import main as m

flags = tf.app.flags

# need to adjust 
flags.DEFINE_string("mode","test","test/train/forward")

# other 

flags.DEFINE_string("out_base_dir", "out", "output base dir, usually the out dir under the main dir")
flags.DEFINE_string("run_id", "0", "the default id is 0" )
flags.DEFINE_string("model_name","basic", "default model name is basic")
flags.DEFINE_string("out_dir","", "")
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")

def main(self):
    config = flags.FLAGS
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))
    m(config)
    pass

if __name__ =='__main__':
	 tf.app.run()

#%%