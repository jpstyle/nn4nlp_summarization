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
from utils import get_avg_loss, batch2input, batch2output


def get_model(model_file_path=None):
    model = Model()
    optimizer = Optimizer(config.optim, config.lr_coverage if config.cov else config.lr,
                          acc=config.adagrad_init_acc, max_grad_norm=config.max_grad_norm)
    optimizer.set_parameters(model.parameters())

    start_iter, start_loss = 0, 0
    if model_file_path is not None:
        checkpoint = torch.load(model_file_path)
        start_iter = checkpoint['iter']
        start_loss = checkpoint['current_loss']

        model_state_dict = dict([(k, v)
                                 for k, v in checkpoint['model'].items()])
        model.load_state_dict(model_state_dict, strict=False)

        if not config.cov:
            optimizer.optim.load_state_dict(checkpoint['optimizer'])
            if config.use_gpu:
                for state in optimizer.optim.state.values():
                    for k, v in checkpoint.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
    if len(config.gpus) > 0:
        model = model.cuda()
        if len(config.gpus) > 1:
            model = nn.DataParallel(model, config.gpus)
        optimizer.set_parameters(model.parameters())
    return model, optimizer, start_iter, start_loss


def trainEpochs(epochs, data, vocab, model_save_dir, model_file_path=None, logger=None):
    def get_train_batches():
        return batchify(data.get_training_examples(), config.batch_size, vocab)

    model, optim, iter, avg_loss = get_model(model_file_path)
    start = time.time()

    for ep in range(epochs):
        batches = get_train_batches()
        for batch in batches:
            optim.zero_grad()
            enc_input, enc_mask, sec_mask, enc_lens, enc_sec_lens, enc_input_oov, zeros_oov, context, coverage = batch2input(batch, len(config.gpus) > 0)
            dec_input, dec_mask, dec_len, dec_lens, target = batch2output(batch, len(config.gpus) > 0)

            loss, pred = model(batch.sec_num, batch.sec_len, enc_input, enc_mask, sec_mask, enc_lens, enc_sec_lens, enc_input_oov, zeros_oov, context, coverage, dec_input, dec_mask, dec_len, dec_lens, target)
            if len(config.gpus) > 1:
                loss = loss.mean()
            if loss.item() != loss.item():
                raise ValueError
            loss.backward()
            optim.step()
            loss = loss.item()
            avg_loss = get_avg_loss(loss, avg_loss, logger, iter)
            
            iter += 1
            if iter % config.print_interval == 0:
                logger.flush()
                time_took = time.time()-start
                start = time.time()
                print(f'epoch {ep} ({iter} steps); loss: {loss:.4f}, time: {time_took:.2f} ({time_took/config.print_interval} s/step)')
                try:
                    pred = pred[0]
                    pred = [int(x) for x in list(pred.cpu().numpy())]
                    print("output: "+" ".join([vocab.get(x, batch.articles[0].oovv.get(x, " ")) for x in pred]))
                    print(f"target: {' '.join(batch.abstracts[0].words)}")
                except:
                    pass

        checkpoint = {
            'model': model.module.state_dict() if len(config.gpus) > 1 else model.state_dict(),
            'config': config,
            'iter': iter,
            'optimizer': optim.optim.state_dict(),
            'loss': avg_loss
        }
        model_save_path = os.path.join(model_save_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(checkpoint, model_save_path)

def main():
    data = DataLoader(config)
    vocab = data.vocab
    logger = tf.summary.FileWriter(config.log_save_dir)
    trainEpochs(config.ep, data, vocab, config.save_dir, config.train_from, logger)

if __name__ == '__main__':
    main()
