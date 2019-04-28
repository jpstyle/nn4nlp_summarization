from __future__ import unicode_literals, print_function, division

import sys
import os
import time
import logging
import pickle

import torch
import pyrouge
from torch.autograd import Variable
from pre_process import batchify
from utils import batch2input, batch2output


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage, focus, betas):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage
    self.focus = focus
    self.betas = betas

  def extend(self, token, log_prob, state, context, coverage, focus):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage,
                      focus = focus,
                      betas = self.betas + [beta])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model, config, data, vocab):
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (config.load_from.split("/")[-1]))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.model = model
        self.vocab = vocab
        self.batcher = batchify(data.get_test_examples(), config.batch_size, vocab, repeat=True)


    def decode(self, config):
        start = time.time()
        counter = 0
        pick = {}

        for batch in self.batcher:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch, config)

            pick[batch.articles[0].id] = {}

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = [self.vocab.get(x, batch.articles[0].oovv.get(x-len(self.vocab), "<???>")) for x in output_ids]

            original_abstract_sents = batch.abstracts[0].words[1:] # Reference summary

            # Remove the [STOP] token if necessary
            try:
                fst_stop_idx = decoded_words.index(self.vocab.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            try:
                fst_stop_idx = original_abstract_sents.index(self.vocab.STOP_DECODING)
                original_abstract_sents = original_abstract_sents[:fst_stop_idx]
            except ValueError:
                original_abstract_sents = original_abstract_sents

            words_all = []
            words = []
            betas_all = []
            betas = []
            for i, w in enumerate(decoded_words):
                if w == self.vocab.EOS:
                    words_all.append(words)
                    betas_all.append(betas)
                    words = []
                    betas = []
                else:
                    words.append(w)
                    betas.append(best_summary.betas[i])
            if len(betas) != 0:
                words_all.append(words)
                betas_all.append(betas)

            pick[batch.articles[0].id]["words"] = words_all
            pick[batch.articles[0].id]["betas_word"] = [[bt.tolist() for bt in bt_st] for bt_st in betas_all]
            pick[batch.articles[0].id]["betas_sent"] = [torch.stack(bt_st).mean(dim=0).tolist() for bt_st in betas_all]

            # Remove EOS markers
            while self.vocab.EOS in decoded_words: decoded_words.remove(self.vocab.EOS)
            while self.vocab.EOS in original_abstract_sents: original_abstract_sents.remove(self.vocab.EOS)

            write_for_rouge(original_abstract_sents, decoded_words, batch.articles[0].id,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 100 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

        pick_file = open(os.path.join(self._decode_dir, "betas.pickle"), 'wb')
        pickle.dump(pick, pick_file)
        pick_file.close()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch, config):
        # Encode input
        enc_input, enc_mask, sec_mask, enc_lens, enc_sec_lens, enc_input_oov, zeros_oov, context, coverage = batch2input(batch, len(config.gpus) > 0)
        enc_outputs, enc_feature, enc_sec_outputs, hidden = self.model.encoder(enc_input, enc_lens, enc_sec_lens, batch.sec_num, batch.sec_len)

        dec_h, dec_c = hidden
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        if config.hard:
            focus = torch.zeros_like(sec_mask)
            focus[:, 0] = 1.0
        else:
            focus = None

        # Initialize hypotheses
        beams = [Beam(tokens=[self.vocab[self.vocab.START_DECODING]],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=context[0],
                      coverage=(coverage[0] if config.cov else None),
                      focus=focus[0] if config.hard else None,
                      betas=[])
                 for _ in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_len and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(self.vocab) else self.vocab[self.vocab.UNK] for t in latest_tokens]

            dec_input = Variable(torch.LongTensor(latest_tokens))
            if len(config.gpus) > 0:
                dec_input = dec_input.cuda()

            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                all_state_h.append(h.state[0])
                all_state_c.append(h.state[1])
                all_context.append(h.context)

            hidden = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            context = torch.stack(all_context, 0)

            coverage = None
            if config.cov:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage = torch.stack(all_coverage, 0)

            focus = None
            if config.hard:
                all_focus = []
                for h in beams:
                    all_focus.append(h.focus)
                focus = torch.stack(all_focus, 0)

            final_dist, hidden, context, attn_dist, _coverage, focus, beta = self.model.decoder(dec_input, hidden, enc_outputs, enc_feature,
                                                    enc_sec_outputs, enc_mask, sec_mask, context, zeros_oov, enc_input_oov, coverage, focus)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = hidden
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                hidden_i = (dec_h[i], dec_c[i])
                context_i = context[i]
                coverage_i = _coverage[i] if config.cov else None
                focus_i = focus[i] if config.hard else None
                beta_i = focus[i] if config.hard else beta[i]

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=hidden_i,
                                   context=context_i,
                                   coverage=coverage_i,
                                   focus=focus_i,
                                   beta=beta_i)
                    all_beams.append(new_beam)

            beams = []
            for h in sort_beams(all_beams):
                if h.latest_token == self.vocab[self.vocab.STOP_DECODING]:
                    if steps >= config.min_dec_len:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = sort_beams(results)

        return beams_sorted[0]

def sort_beams(beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


def write_for_rouge(reference_sents, decoded_words, ex_index, _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    ref_file = os.path.join(_rouge_ref_dir, "%s_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%s_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155("./ROUGE-1.5.5")
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\w+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir

    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1","2","3","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..."%(results_file))
    with open(results_file, "w") as f:
        f.write(log_str)