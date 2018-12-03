#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:37:45 2018

@author: xavier.qiu
"""
import argparse
import json
import os
from collections import Counter
from tqdm import tqdm
from squad.utils import get_word_span, get_word_idx, process_tokens

def main():
    args = get_args()
    prepro(args)
    
def get_args( ):
    pass
    
def prepro(args):
    pass