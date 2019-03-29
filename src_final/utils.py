import numpy as np
import torch
from config import config
import tensorflow as tf

def get_input_from_batch(batch, use_cuda):
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
    enc_padding_mask = enc_batch.ne(0).float().requires_grad_()
    sec_padding_mask = torch.LongTensor(sec_padding_mask).float().requires_grad_()

    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    batch_size = len(batch)
    if config.pointer_gen:
        enc_batch_extend_vocab = torch.LongTensor(inputs_oov)
    if batch.max_oov > 0:
        extra_zeros = torch.zeros((batch_size, batch.max_oov), requires_grad=True)

    context = torch.zeros((batch_size, 2 * config.hidden_dim),requires_grad=True)

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size(),requires_grad=True)

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()
        c_t_1 = c_t_1.cuda()
        sec_padding_mask = sec_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, sec_padding_mask, enc_lens, batch.sec_lens, enc_batch_extend_vocab, extra_zeros, context, coverage

def get_output_from_batch(batch, use_cuda):
    targets = []
    targets_oov = []
    for abstract in batch.abstracts:
        targets.append(abstract.word_ids)
        targets_oov.append(abstract.word_ids_oov)

    dec_batch = torch.LongTensor(targets)
    dec_padding_mask = dec_batch.ne(0).float().requires_grad_()
    dec_lens = batch.dec_lens
    max_dec_len = max(dec_lens)
    dec_lens_var = torch.Tensor(dec_lens).float().requires_grad_()
    target_batch = torch.LongTensor(targets_oov)

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss


def initialize_lstm(lstm):
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
    weight.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

  for b in biases:
    bias = getattr(lstm, b)
    bias.data.fill_(0.)
    bias.data[(bias.size(0) // 4):(bias.size(0) // 2)].fill_(1.)


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