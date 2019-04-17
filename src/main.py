from __future__ import unicode_literals, print_function, division

# import ptvsd
# ptvsd.enable_attach(address=('localhost', 99), redirect_output=True)
# ptvsd.wait_for_attach()


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

    model, optim, iter, avg_loss = get_model(model_file_path)
    start = time.time()

    for ep in range(epochs):
        batches = get_train_batches()
        for batch in batches:
            optim.zero_grad()
            loss, pred = model(batch)
            loss.backward()
            optim.step()
            loss = loss.item()
            avg_loss = get_avg_loss(loss, avg_loss, logger, iter)
            
            iter += 1
            if iter % config.print_interval == 0:
                logger.flush()
                time_took = time.time()-start
                start = time.time()
                print(f'epoch {ep} ({iter} steps); loss: {loss:.4f}, time: {time_took:.2f} ({time_took/config.print_interval} step/s)')
                try:
                    print("output: "+" ".join([vocab.get(x, batch.articles[0].oovv.get(x, " ")) for x in pred]))
                    print(f"target: {' '.join(batch.abstracts[0].words)}")
                except:
                    pass
            if iter % config.save_interval == 0:
                checkpoint = {
                    'model': model.module.state_dict() if len(config.gpus) > 1 else model.state_dict(),
                    'config': config,
                    'iter': iter,
                    'optimizer': optim.optim.state_dict(),
                    'loss': avg_loss
                }
                model_save_path = os.path.join(model_save_dir, 'model_%d_%d' % (iter, int(time.time())))
                torch.save(checkpoint, model_save_path)

def main(model_path):
    data = DataLoader(config)
    vocab = data.vocab
    logger = tf.summary.FileWriter(config.log_save_dir)
    trainEpochs(config.ep, data, vocab, config.save_dir, model_path, logger)

if __name__ == '__main__':
    main(config.train_from)
