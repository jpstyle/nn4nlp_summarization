import numpy as np
import torch
from config import config
import tensorflow as tf


def batch2input(batch, use_cuda):
    inputs = []
    inputs_oov = []
    sec_padding_mask = []
    for article in batch.articles:
        inp = []
        inp_oov = []
        for sec in article.secs:
            inp.extend(sec.word_ids)
            inp_oov.extend(sec.word_ids_oov)
        inputs.append(inp)
        inputs_oov.append(inp_oov)
        sec_padding_mask.append(article.sec_mask)

    enc_batch = torch.LongTensor(inputs)
    enc_padding_mask = enc_batch.ne(0).float()
    sec_padding_mask = torch.LongTensor(sec_padding_mask).float()

    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    batch_size = len(batch)
    if config.pointer:
        enc_batch_extend_vocab = torch.LongTensor(inputs_oov)
    if batch.max_oov > 0:
        extra_zeros = torch.zeros((batch_size, batch.max_oov))

    context = torch.zeros(
        (batch_size, 2 * config.hidden_dim), requires_grad=True)

    coverage = None
    if config.cov:
        coverage = torch.zeros(enc_batch.size(), requires_grad=True)

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()
        sec_padding_mask = sec_padding_mask.cuda()
        context = context.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, sec_padding_mask, enc_lens, batch.sec_lens, enc_batch_extend_vocab, extra_zeros, context, coverage


def batch2output(batch, use_cuda):
    targets = []
    targets_oov = []
    for abstract in batch.abstracts:
        targets.append(abstract.word_ids)
        targets_oov.append(abstract.word_ids_oov)

    dec_input = torch.LongTensor(targets)[:,:-1]
    dec_mask = dec_input.ne(0).float()
    dec_lens = batch.dec_lens
    dec_len = max(dec_lens)
    dec_lens = torch.Tensor(dec_lens).float()
    target = torch.LongTensor(targets_oov)[:,1:]

    if use_cuda:
        dec_input = dec_input.cuda()
        dec_mask = dec_mask.cuda()
        dec_lens = dec_lens.cuda()
        target = target.cuda()

    return dec_input, dec_mask, dec_len, dec_lens, target


def get_avg_loss(loss, avg_loss, summary_writer, step, decay=0.99):
    if loss == 0:  # on the first iteration just take the loss
        avg_loss = loss
    else:
        avg_loss = avg_loss * decay + (1 - decay) * loss
    avg_loss = min(avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return avg_loss


def init(lstm):
    weights = []
    biases = []
    for wns in lstm._all_weights:
        for wn in wns:
            if wn.startswith('weight'):
                weights.append(wn)
            elif wn.startswith('bias'):
                biases.append(wn)

    for w in weights:
        weight = getattr(lstm, w)
        weight.data.uniform_(-config.rand_unif_init_mag,
                             config.rand_unif_init_mag)

    for b in biases:
        bias = getattr(lstm, b)
        bias.data.fill_(0.)
        bias.data[(bias.size(0) // 4):(bias.size(0) // 2)].fill_(1.)
