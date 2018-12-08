import argparse
import json
import math
import os
import shutil
from pprint import pprint

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.read_data import read_data, get_squad_data_filter, update_config
from basic.trainer import MultiGPUTrainer


def main(config):
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)


def set_dirs(config):
    pass


def _config_debug(config):
    pass


def _train(config):
    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    dev_data = read_data(config, 'dev', config.load, data_filter=data_filter)
    update_config(config, [train_data, dev_data])
    _config_debug(config)

    if config.lower_word:
        word2vec_dict = train_data.shared['lower_word2vec']
    else:
        word2vec_dict = train_data.shared['word2vec']

    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict else np.random.multivariate_normal(
        np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    pprint(config.flag_values_dict(), indent=2)

    models = get_multi_gpu_models(config)
    model = models[0]
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    graph_handler.initialize(sess)

    num_steps = config.num_steps or int(
        math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0

    for batches in tqdm(train_data.get_multi_batches(batch_size=config.batch_size,
                                                     num_batches_per_step=config.num_gpus,
                                                     num_steps=num_steps,
                                                     shuffle=True,
                                                     cluster=config.cluster),
                        total=num_steps):
        global_step = sess.run(model.global_step) + 1

        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)
        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            if 0 < config.val_num_batches < num_steps:
                num_steps = config.val_num_batches
            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps),
                           total=num_steps)
            )
            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps),
                           total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)

            if config.dump_eval:
                graph_handler.dump_eval(e_dev)
            if config.dump_answer:
                graph_handler.dump_answer(e_dev)
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)


def _test(config):
    pass
