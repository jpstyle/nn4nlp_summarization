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

class Train(object):
    def __init__(self):
        self.data = DataLoader(config)
        self.vocab = self.data.vocab

        train_dir = os.path.join(config.log_root,config.save_dir+"_"+str(int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join("models",config.save_dir+"_"+str(int(time.time())))
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def get_train_batches(self):
        return batchify(self.data.get_training_examples(), config.batch_size, self.vocab)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)
        if config.gpus:
            self.model = self.model.cuda()

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Optimizer(config.optim, config.lr_coverage if config.is_coverage else config.lr, acc=config.adagrad_init_acc, max_grad_norm=config.max_grad_norm)
        self.optimizer.set_parameters(params)
        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.optim.load_state_dict(state['optimizer'])
                if config.use_gpu:
                    for state in self.optimizer.optim.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def trainEpochs(self, epochs, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()

        for ep in range(epochs):
            batches = self.get_train_batches()
            for batch in batches:
                self.optimizer.zero_grad()
                loss, pred = self.model(batch)
                loss.backward()
                self.optimizer.step()
                loss = loss.item()

                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
                iter += 1

                if iter % 100 == 0:
                    self.summary_writer.flush()

                if iter % config.print_interval == 0:
                    print('steps %d, loss: %f,  seconds for %d batch: %.2f' % (iter, loss, config.print_interval,
                                                                            time.time() - start))
                    print("output: "+" ".join([self.vocab.get(x, batch.articles[0].oovv.get(x, " ")) for x in pred]))
                    print(f"target: {' '.join(batch.abstracts[0].words)}")
 
                    start = time.time()
                if iter % config.save_interval == 0:
                    self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    train_processor = Train()
    train_processor.trainEpochs(config.max_iterations, config.train_from)
