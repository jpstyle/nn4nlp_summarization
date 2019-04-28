from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from numpy import random
from utils import init, batch2input, batch2output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=0)
        self.embedding.weight.data.normal_(std=config.trunc_norm_init_std)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        self.lstm_sec = nn.LSTM(config.hidden_dim, config.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        init(self.lstm)
        init(self.lstm_sec)

        self.w_sec = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w_feat = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_h.weight.data.normal_(std=config.trunc_norm_init_std)
        self.reduce_h.bias.data.normal_(std=config.trunc_norm_init_std)

        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c.weight.data.normal_(std=config.trunc_norm_init_std)
        self.reduce_c.bias.data.normal_(std=config.trunc_norm_init_std)

    def _reduce_state(self, hidden):
        h, c = hidden
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in)).unsqueeze(0)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in)).unsqueeze(0)
        return (hidden_reduced_h, hidden_reduced_c)

    def forward(self, input, seq_lens, sec_lens, secL, wordL):
        b, l = input.shape
        embedded = self.embedding(input)

        section_encodings = []
        section_reps = []

        for i in range(secL):
            packed = nn.utils.rnn.pack_padded_sequence(embedded[:,i*wordL:(i+1)*wordL,:], [wordL]*b, batch_first=True)
            self.lstm.flatten_parameters()
            output, hidden = self.lstm(packed)
            enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            enc_outputs = enc_outputs.contiguous()

            section_encodings.append(enc_outputs)
            section_reps.append(self.w_sec(hidden[0].view(-1, 2*config.hidden_dim)))

        enc_outputs = torch.cat(section_encodings, dim=1)
        sec_input = torch.stack(section_reps, dim=1)

        enc_feature = enc_outputs.view(b, -1, 2*config.hidden_dim)
        enc_feature = self.w_feat(enc_feature)

        packed_sec = nn.utils.rnn.pack_padded_sequence(sec_input, [secL]*b, batch_first=True)
        self.lstm_sec.flatten_parameters()
        output, hidden = self.lstm_sec(packed_sec)

        enc_sec_outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        enc_sec_outputs = enc_sec_outputs.contiguous()

        hidden = self._reduce_state(hidden)

        return enc_outputs, enc_feature, enc_sec_outputs, hidden


class SectionAttention(nn.Module):
    def __init__(self):
        super(SectionAttention, self).__init__()
        self.w_d = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.w_feat = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, dec_hidden, enc_sec_outputs, sec_mask):
        batch, secL, dim = enc_sec_outputs.shape

        enc_sec_feature = self.w_feat(enc_sec_outputs.view(-1, 2*config.hidden_dim))
        dec_feature = self.w_d(dec_hidden).unsqueeze(1).expand(batch, secL, dim).contiguous().view(-1, dim)
        att_features = enc_sec_feature + dec_feature
        score = self.v(torch.tanh(att_features)).view(-1, secL)

        attn_dist_ = self.softmax(score)*sec_mask
        attn_dist = attn_dist_ / attn_dist_.sum(1, keepdim=True)
        return attn_dist

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        if config.cov:
            self.w_cov = nn.Linear(1, config.hidden_dim * 2, bias=False)

    def forward(self, dec_hidden, enc_output, enc_feature, enc_mask, sec_attn, coverage):
        batch, src_len, dim = enc_output.shape

        dec_feature = self.decode_proj(dec_hidden).unsqueeze(1).expand(batch, src_len, dim).contiguous()
        dec_feature = dec_feature.view(batch, -1, dim)

        att_features = enc_feature + dec_feature
        if config.cov:
            coverage_input = coverage.view(batch, -1, 1)
            coverage_feature = self.w_cov(coverage_input)
            att_features = att_features + coverage_feature

        score = self.v(torch.tanh(att_features)).view(-1, src_len)

        secL = sec_attn.size(1)
        wordL = int(src_len/secL)
        sec_attn = sec_attn.unsqueeze(2).repeat(1,1,wordL).view(batch,-1)

        attn_dist = torch.einsum("bl,bl->bl",sec_attn, score)
        attn_dist = self.softmax(score)*enc_mask
        attn_dist = attn_dist / attn_dist.sum(1, keepdim=True)

        attn_dist = attn_dist.unsqueeze(1)
        context = torch.bmm(attn_dist, enc_output).view(-1, config.hidden_dim * 2)
        attn_dist = attn_dist.view(-1, src_len)
        if config.cov:
            coverage = coverage.view(-1, src_len) + attn_dist

        return context, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=0)
        self.attn = Attention()
        self.attn_sec = SectionAttention()
        self.combine_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init(self.lstm)

        if config.pointer:
            self.generation_p = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        self.out = nn.Linear(config.hidden_dim * 3, config.vocab_size)
        self.out.weight.data.normal_(std=config.trunc_norm_init_std)
        self.out.bias.data.normal_(std=config.trunc_norm_init_std)

    def forward(self, inp, hidden, enc_outputs, enc_feature, enc_sec_output, enc_mask, sec_mask, 
                prev_context, zeros_oov, enc_input_oov, coverage):

        inp = self.embedding(inp)
        inp = self.combine_context(torch.cat((prev_context, inp), 1))
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(inp.unsqueeze(1), hidden)

        dec_state = torch.cat((hidden[0].view(-1, config.hidden_dim), hidden[1].view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        sec_attn_dist = self.attn_sec(dec_state, enc_sec_output, sec_mask)
        context, attn_dist, coverage = self.attn(dec_state, enc_outputs, enc_feature, enc_mask, sec_attn_dist, coverage)

        if config.pointer:
            p_gen_input = torch.cat((context, dec_state, inp), 1)
            p_gen = torch.sigmoid(self.generation_p(p_gen_input))
        else:
            p_gen = None

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), context), 1)
        output = self.out(output)
        vocab_dist = self.softmax(output)

        if config.pointer:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            if zeros_oov is not None:
                vocab_dist_ = torch.cat((vocab_dist_, zeros_oov), 1)
            final_dist = vocab_dist_.scatter_add(1, enc_input_oov, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, hidden, context, attn_dist, coverage, sec_attn_dist

class Model(nn.Module):
    def __init__(self, tie_emb=True):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if tie_emb:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, sec_num, sec_len, enc_input, enc_mask, sec_mask, enc_lens, enc_sec_lens, enc_input_oov, zeros_oov, context, coverage, dec_input, dec_mask, dec_len, dec_lens, target):
        
        enc_outputs, enc_feature, enc_sec_outputs, hidden = self.encoder(enc_input, enc_lens, enc_sec_lens, sec_num, sec_len)

        losses, preds = [], []
        for t in range(min(dec_len, config.max_dec_len)-1):
            inputs = dec_input[:, t]
            step_target = target[:, t].unsqueeze(1)
            final_dist, hidden, context, attn_dist, _coverage, _ = self.decoder(inputs, hidden, enc_outputs, enc_feature, enc_sec_outputs, enc_mask, sec_mask, context,
                                                        zeros_oov, enc_input_oov, coverage)
            preds.append(final_dist[0].argmax().item())
            target_prob = torch.gather(final_dist, 1, step_target).squeeze() + config.eps
            loss_t = -torch.log(target_prob)
            if config.cov:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                loss_t = loss_t + config.cov_loss_wt * step_coverage_loss
                coverage = _coverage
                
            loss_t = loss_t * dec_mask[:, t]
            losses.append(loss_t)

        losses = torch.sum(torch.stack(losses, 1), 1)
        loss = torch.mean(losses/dec_lens)
            
        return loss, torch.tensor(preds).unsqueeze(0).to(loss.device)
