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
from squad.utils_cq import get_word_span, get_word_idx, process_tokens
import nltk


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    target_dir = "/Users/xavier.qiu/Documents/ricecourse/bidaf/data/"
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-s', "--source_dir", default=target_dir)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--version", default="1.1", type=str)
    return parser.parse_args()
    pass


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def prepro_each(args, dataset_type, out_name, start_ratio=0.0, end_ratio=1.0, in_path=None):
    """
    
    :param args: 
    :param dataset_type: train or dev,
    :param out_name: 
    :param start_ratio: 
    :param end_ratio: 
    :param in_path: 
    :return: 
    """

    sent_tokenize = nltk.sent_tokenize

    def word_tokenize(tokens):
        """

        :param tokens:
        :return:
        """
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    if not args.split:
        sent_tokenize = lambda para: [para]
        pass

    # 1. load data
    source_path = in_path or os.path.join(args.source_dir, "{}-v{}.json".format(dataset_type, args.version))
    print("loading data from {} ... ... [train-v1.1.json]".format(source_path))
    source_data = json.load(open(source_path, 'r'))

    word_counter = Counter()
    lower_word_counter = Counter()
    char_counter = Counter()

    article_number = len(source_data['data'])
    print("there are {} articles in data set - {}".format(article_number, dataset_type))
    start_at_index = int(round(article_number * start_ratio))
    end_at_index = int(round(article_number * end_ratio))

    article_paragraph_sentence_wordlist = []  # x
    article_paragraph_sentence_word_charlist = []  # cx
    article_contextlist = []  # pp

    question_wordlist = []
    question_word_charlist = []
    question_answertextlist = []

    for article_index, article in enumerate(tqdm(source_data['data'][start_at_index: end_at_index])):

        paragraph_sentence_wordlist = []
        paragraph_sentence_word_charlist = []
        contextlist = []

        article_paragraph_sentence_wordlist.append(paragraph_sentence_wordlist)
        article_paragraph_sentence_word_charlist.append(paragraph_sentence_word_charlist)
        article_contextlist.append(contextlist)

        for paragraph_index, paragraph in enumerate(article['paragraphs']):
            # clean context
            context = paragraph['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            # tokenize into word and char lists
            sentence_wordlist = list(map(word_tokenize, sent_tokenize(context)))  # 2d, a list of wordlist
            sentence_wordlist = [process_tokens(tokens) for tokens in sentence_wordlist]
            # process tokens will do this ["wgw/gweg", "fwe~few"] -> ['wgw', '/', 'gweg', 'fwe', '~', 'few']

            sentence_word_charlist = [[list(word) for word in wordlist] for wordlist in
                                      sentence_wordlist]  # 3d, a list of word, and each word is a char list

            # append them into the data
            paragraph_sentence_wordlist.append(sentence_wordlist)
            paragraph_sentence_word_charlist.append(sentence_word_charlist)
            contextlist.append(context)

            # counter words and chars
            number_qas = len(paragraph['qas'])
            for wordlist in sentence_wordlist:
                for word in wordlist:
                    word_counter[word] += number_qas
                    lower_word_counter[word.lower()] += number_qas
                    for char in word:
                        char_counter[char] += number_qas

            rxi = [article_index, paragraph_index]
            assert len(article_paragraph_sentence_wordlist) - 1 == article_index
            assert len(article_paragraph_sentence_wordlist[article_index]) - 1 == paragraph_index

            for question in paragraph['qas']:
                wordlist = word_tokenize(question['question'])
                word_charlist = [list(word) for word in wordlist]

                answer_index_pair = []
                answer_textlist = []

                for answer in question['answers']:
                    answer_text = answer['text']
                    answer_textlist.append(answer_text)

                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)

                    yi0, yi1 = get_word_span(context, sentence_wordlist, answer_start, answer_stop)

                    assert len(sentence_wordlist[yi0[0]]) > yi0[1]

                for word in wordlist:
                    word_counter[word] += 1
                    lower_word_counter[word.lower()] += 1
                    for char in word:
                        char_counter[char] += 1
                pass
            pass
        pass
    pass
