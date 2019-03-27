import torch
import os
import logging
import datetime
import codecs

from torch.autograd import Variable
import numpy as np
import torch
import config

def get_input_from_batch(batch, use_cuda):
    # batch_size = len(batch.enc_lens)
    batch_size = len(batch)

    inputs = []
    inputs_oov = []
    sec_padding_mask = []
    for article in batch.articles:
        inp = []
        inp_oov = []
        for sec in article.secs:
            # inp.append(sec.word_ids) # for hierarchical
            # inp_oov.append(sec.word_ids_oov)
            inp.extend(sec.word_ids)
            inp_oov.extend(sec.word_ids_oov)
        inputs.append(inp)
        inputs_oov.append(inp_oov)
        sec_padding_mask.append(article.sec_mask)

    enc_batch = torch.LongTensor(inputs)
    # enc_batch = torch.Tensor(torch.from_numpy(batch.enc_batch).long(), requires_grad=True)
    # enc_padding_mask = torch.Tensor(torch.from_numpy(batch.enc_padding_mask), requires_grad=True).float()
    enc_padding_mask = enc_batch.ne(0).float().requires_grad_()
    # torch.Tensor(torch.from_numpy(batch.enc_padding_mask)).float()

    sec_padding_mask = torch.LongTensor(sec_padding_mask).float().requires_grad_()

    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = torch.LongTensor(inputs_oov)
        # enc_batch_extend_vocab = torch.Tensor(torch.from_numpy(batch.enc_batch_extend_vocab, requires_grad=True).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_oov > 0:
        extra_zeros = torch.zeros((batch_size, batch.max_oov), requires_grad=True)
        # extra_zeros = torch.Tensor(torch.zeros((batch_size, batch.max_oov)), requires_grad=True)

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim),requires_grad=True)

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

    return enc_batch, enc_padding_mask, sec_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, use_cuda):
    targets = []
    targets_oov = []
    for abstract in batch.abstracts:
        # targets.append([vocab[vocab.START_DECODING]] + abstract.word_ids + [vocab[vocab.STOP_DECODING]])
        # targets_oov.append([vocab[vocab.START_DECODING]] + abstract.word_ids_oov + [vocab[vocab.STOP_DECODING]])
        targets.append(abstract.word_ids)
        targets_oov.append(abstract.word_ids_oov)

    dec_batch = torch.LongTensor(targets)
    dec_padding_mask = dec_batch.ne(0).float().requires_grad_()
    dec_lens = batch.dec_lens
    max_dec_len = max(dec_lens)
    dec_lens_var = torch.Tensor(dec_lens).float().requires_grad_()
    # dec_lens_var = torch.Tensor(torch.from_numpy(dec_lens), requires_grad=True).float()
    target_batch = torch.LongTensor(targets_oov)
    # target_batch = torch.Tensor(torch.from_numpy(batch.target_batch), requires_grad=True).long()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch



def import_fasttext(vocab, embFile, pad_idx=0):
    import numpy as np
    dim = 300
    vocab_keys = list(vocab._w2i.keys())
    vecs = np.zeros((len(vocab), dim))
    vecs_check = [False] * len(vocab)
    with codecs.open(embFile, "r", "utf-8") as f:
        for l in f:
            items = l.strip().split()
            if len(items) < 301:
                continue
            try:
                v = np.array(items[1:], dtype=np.float32)
            except Exception as e:
                print (items)
                continue
            key = items[0]
            if key in vocab_keys:
                vecs[vocab.get(key)] = v
                vecs_check[vocab.get(key)] = True
    print(f"{sum(vecs_check)} vectors are loaded from fast-text embedding")
    for i,vec_bool in enumerate(vecs_check):
        if i == pad_idx:
            continue
        if not vec_bool:
            vecs[i] = np.random.normal(size=dim)
    embeddings = torch.Tensor(vecs)
    torch.save(embeddings, "data/fasttext.pt")
    return embeddings

def save_results(path, train_losses, val_losses, best_val_loss, best_ep):
    res_dict = {}
    res_dict["best_val_loss"] = best_val_loss
    res_dict["best_ep"] = best_ep
    res_dict["train"] = train_losses
    res_dict["loss"] = val_losses
    with open(path, "w") as file:
        file.write(str(res_dict))

def plot_loss_figure(path, train, val, start_ep=1):
    """Plot training loss points (maximum four sets of points) and save as a figure."""
    data = [train, val]
    legends = ["train_loss", "val_loss"]
    colors = ["navy","orangered"]
    
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    eps = list(range(start_ep+1, start_ep+len(train)+1))

    fig = plt.figure()
    fig, ax = plt.subplots()
    for i, points in enumerate(data):
        plt.plot(eps, points, label=legends[i], color=colors[i])

    ax.set_title("train/val loss")
    plt.xlabel('Epochs')
    plt.ylabel("loss")
    plt.legend()
    fig.savefig(path)
    plt.close(fig)


def test_batcher(batch):
    # code below is for testing
    abs_num_sents = []
    abs_sent_len = []

    art_num_secs = []
    art_num_sents = []
    art_sent_len = []

    for abstract in batch.abstracts:
        abs_num_sents += [len(abstract.sents)]
        for sent in abstract.sents:
            abs_sent_len += [len(sent)]
    for article in batch.articles:
        art_num_secs += [len(article.secs)]
        for sec in article.secs:
            art_num_sents += [len(sec.sents)]
            for sent in sec.sents:
                art_sent_len += [len(sent)]

    assert len(set(abs_num_sents)) == 1
    assert len(set(abs_sent_len)) == 1
    assert len(set(art_num_secs)) == 1
    assert len(set(art_num_sents)) == 1
    assert len(set(art_sent_len)) == 1

def get_inputs(articles, oov=False, cuda=False):
    '''convert batch into a input tensor (secLen * wordLen * batch )'''
    inputs = []
    inputs_oov = []
    for article in articles:
        inp = []
        inp_oov = []
        for sec in article.secs:
            inp.append(sec.word_ids)
            inp_oov.append(sec.word_ids_oov)
        inputs.append(inp)
        inputs_oov.append(inp_oov)
    inputs = torch.LongTensor(inputs)
    if oov:
        inputs_oov = torch.LongTensor(inputs_oov)
    else:
        inputs_oov = None
   
    if cuda:
        with torch.cuda.device(0):
            inputs = inputs.cuda()
            if oov:
                inputs_oov = inputs_oov.cuda()
    return inputs, inputs_oov

def get_targets(abstracts, vocab, oov=False, cuda=False):
    '''convert batch into a target tensor (wordLen*batch)'''
    targets = []
    targets_oov = []
    for abstract in abstracts:
        # targets.append([vocab[vocab.START_DECODING]] + abstract.word_ids + [vocab[vocab.STOP_DECODING]])
        # targets_oov.append([vocab[vocab.START_DECODING]] + abstract.word_ids_oov + [vocab[vocab.STOP_DECODING]])
        targets.append(abstract.word_ids)
        targets_oov.append(abstract.word_ids_oov)
    targets = torch.LongTensor(targets)
    if oov:
        targets_oov = torch.LongTensor(targets_oov)
    else:
        targets_oov = None
    
    if cuda:
        with torch.cuda.device(0):
            targets = targets.cuda()
            if oov:
                targets_oov = targets_oov.cuda()
    return targets, targets_oov


def get_logger(name, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', "%m-%d %H:%M:%S")

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    now = datetime.datetime.now()
    file_handler = logging.FileHandler(f'{save_dir}/{name}_{now.month}{now.day}_{now.hour}{now.minute}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger