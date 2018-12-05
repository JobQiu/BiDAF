#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:16:04 2018

@author: xavier.qiu
"""
# test 1
para = "this is a test paragraph, baba lala, test is boring, hope you like this one, nope, \n what about this one?"

import nltk
sent_tokenize = nltk.sent_tokenize
sent_tokenize = lambda para:[para]

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``",'"') for token in nltk.word_tokenize(tokens)]


sent_tokenize(para)
#%%
# test 2
list_of_word = list(map(word_tokenize, sent_tokenize(para)))
list_of_word
#%%len(list_of_word)
# test 3
# language list
language = ['French', 'English', 'German']

# another list of language
language1 = ['Spanish', 'Portuguese']

language.extend(language1)

# Extended List
print('Language List: ', language)

#%%
# test 4 
l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        

"([{}])".format("".join(l))
import re
testword = ["wgw/gweg", "fwe~few"]
tokens = []
for token in testword:
    tokens.extend(re.split("([{}])".format("".join(l)),token))
tokens

#%%