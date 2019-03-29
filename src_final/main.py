from __future__ import unicode_literals, print_function, division

import os
import time
import torch
import tensorflow as tf
from torch import nn
from model import Model
from optim import Optimizer
from config import config
from pre_process import DataLoader, batchify, Vocab
from utils import get_model, calc_running_avg_loss


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
                checkpoint = {
                    'model': model.module.state_dict() if len(config.gpus) > 1 else model.state_dict(),
                    'config': config,
                    'iter': iter,
                    'optimizer': optim.optim.state_dict(),
                    'loss': running_avg_loss
                }
                model_save_path = os.path.join(model_save_dir, 'model_%d_%d' % (iter, int(time.time())))
                torch.save(checkpoint, model_save_path)

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
