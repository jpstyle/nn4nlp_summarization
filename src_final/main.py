from __future__ import unicode_literals, print_function, division

# import ptvsd
# ptvsd.enable_attach(address=('localhost', 99), redirect_output=True)
# ptvsd.wait_for_attach()

import os
import time
import argparse

import tensorflow as tf
import torch
from torch import nn
from model import Model
from optim import Optimizer
from config import config
from pre_process import DataLoader, batchify, Vocab
from utils import calc_running_avg_loss


def save(model, optimizer, model_save_dir, running_avg_loss, iter):
    checkpoint = {
        'model': model.module.state_dict() if len(config.gpus) > 1 else model.state_dict(),
        'config': config,
        'iter': iter,
        'optimizer': optimizer.optim.state_dict(),
        'current_loss': running_avg_loss
    }
    model_save_path = os.path.join(model_save_dir, 'model_%d_%d' % (iter, int(time.time())))
    torch.save(checkpoint, model_save_path)

def get_model(model_file_path=None):
    model = Model()
    optimizer = Optimizer(config.optim, config.lr_coverage if config.is_coverage else config.lr, acc=config.adagrad_init_acc, max_grad_norm=config.max_grad_norm)
    optimizer.set_parameters(model.parameters())

    start_iter, start_loss = 0, 0
    if model_file_path is not None:
        checkpoint = torch.load(model_file_path)
        start_iter = checkpoint['iter']
        start_loss = checkpoint['current_loss']

        model_state_dict = dict([(k, v)
                              for k, v in checkpoint['model'].items()])
        model.load_state_dict(model_state_dict, strict=False)

        if not config.is_coverage:
            optimizer.optim.load_state_dict(checkpoint['optimizer'])
            if config.use_gpu:
                for state in optimizer.optim.state.values():
                    for k, v in checkpoint.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
    if config.use_gpu:
        model = model.cuda()
        optimizer.set_parameters(model.parameters())
    return model, optimizer, start_iter, start_loss

def trainEpochs(epochs, data, vocab, model_save_dir, model_file_path=None, logger=None):
    def get_train_batches():
        return batchify(data.get_training_examples(), config.batch_size, vocab)

    model, optim, iter, running_avg_loss = get_model(model_file_path)
    start = time.time()

    for ep in range(epochs):
        batches = get_train_batches()
        for batch in batches:
            optim.zero_grad()
            loss, pred = model(batch)
            loss.backward()
            optim.step()
            loss = loss.item()

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, logger, iter)
            iter += 1
            if iter % config.print_interval == 0:
                logger.flush()
                print('steps %d, loss: %f,  seconds for %d batch: %.2f' % (iter, loss, config.print_interval,
                                                                        time.time() - start))
                print("output: "+" ".join([vocab.get(x, batch.articles[0].oovv.get(x, " ")) for x in pred]))
                print(f"target: {' '.join(batch.abstracts[0].words)}")

                start = time.time()
            if iter % config.save_interval == 0:
                save(model, optim, model_save_dir, running_avg_loss, iter)

def main(model_path):
    data = DataLoader(config)
    vocab = data.vocab

    model_save_dir = os.path.join("models",config.save_dir+"_"+str(int(time.time())))
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    log_save_dir = os.path.join(config.log_root,config.save_dir+"_"+str(int(time.time())))
    if not os.path.isdir(log_save_dir):
        os.mkdir(log_save_dir)
    logger = tf.summary.FileWriter(log_save_dir)

    trainEpochs(config.ep, data, vocab, model_save_dir, model_path, logger)

if __name__ == '__main__':
    main(config.train_from)
