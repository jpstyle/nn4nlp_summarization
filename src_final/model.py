from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import config
from numpy import random
from utils import initialize_lstm, get_input_from_batch, get_output_from_batch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embedding.weight.data.normal_(std=config.trunc_norm_init_std)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        self.lstm_sec = nn.LSTM(config.hidden_dim, config.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        initialize_lstm(self.lstm)
        initialize_lstm(self.lstm_sec)

        self.W_sec = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_h.weight.data.normal_(std=config.trunc_norm_init_std)
        self.reduce_h.bias.data.normal_(std=config.trunc_norm_init_std)

        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c.weight.data.normal_(std=config.trunc_norm_init_std)
        self.reduce_c.bias.data.normal_(std=config.trunc_norm_init_std)

    def _reduce_state(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

    #seq_lens should be in descending order
    def forward(self, input, seq_lens, sec_lens, secL, wordL):
        b, l = input.shape
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        sec_input = encoder_outputs.view(b, secL, wordL, -1)[:,:,-1,:]
        sec_input = self.W_sec(sec_input)
        packed_sec = pack_padded_sequence(sec_input, sec_lens, batch_first=True)
        output, hidden = self.lstm_sec(packed_sec)
        encoder_sec_outputs, _ = pad_packed_sequence(output, batch_first=True)
        encoder_sec_outputs = encoder_sec_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        hidden = self._reduce_state(hidden)

        return encoder_outputs, encoder_feature, encoder_sec_outputs, hidden


class SectionAttention(nn.Module):
    def __init__(self):
        super(SectionAttention, self).__init__()
        # section-level attention
        self.w_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.w_d = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_section_outputs, sec_padding_mask):
        batch, secL, dim = encoder_section_outputs.shape

        encoder_section_feature = encoder_section_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_section_feature = self.w_h(encoder_section_feature)

        dec_fea = self.w_d(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(batch, secL, dim).contiguous() # B x secL x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, dim)  # B * secL x 2*hidden_dim

        att_features = encoder_section_feature + dec_fea_expanded # B * secL x 2*hidden_dim
        e = torch.tanh(att_features) # B * secL x 2*hidden_dim
        scores = self.v(e)  # B * secL x 1
        scores = scores.view(-1, secL)  # B x secL

        attn_dist_ = F.softmax(scores, dim=1)*sec_padding_mask # B x secL
        attn_dist = attn_dist_ / attn_dist_.sum(1, keepdim=True) # B x secL
        return attn_dist

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, sec_attn, coverage):
        batch, src_len, dim = encoder_outputs.shape

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(batch, src_len, dim).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, dim)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, src_len)  # B x t_k

        secL = sec_attn.size(1)
        wordL = int(src_len/secL)
        sec_attn = sec_attn.unsqueeze(2).repeat(1,1,wordL).view(batch,-1) # B x t_k

        attn_dist_ = torch.einsum("bl,bl->bl",sec_attn, scores)
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        attn_dist = attn_dist_ / attn_dist_.sum(1, keepdim=True)

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, src_len)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, src_len)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.attention_network = Attention()
        self.attention_section_network = SectionAttention()
        self.combine_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        initialize_lstm(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out1.weight.data.normal_(std=config.trunc_norm_init_std)
        self.out1.bias.data.normal_(std=config.trunc_norm_init_std)

        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        self.out2.weight.data.normal_(std=config.trunc_norm_init_std)
        self.out2.bias.data.normal_(std=config.trunc_norm_init_std)

    def forward(self, inp, hidden, encoder_outputs, encoder_feature, encoder_sec_outputs, enc_padding_mask, sec_padding_mask, 
                prev_context, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h, c = hidden
            dec_state = torch.cat((h.view(-1, self.hidden_dim), c.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            sec_attn_dist = self.attention_section_network(dec_state, encoder_sec_outputs, sec_padding_mask)
            c_t, _, coverage_next = self.attention_network(dec_state, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, sec_attn_dist, coverage)
            coverage = coverage_next

        inp = self.embedding(inp)
        inp = self.combine_context(torch.cat((prev_context, inp), 1))
        lstm_out, hidden = self.lstm(inp.unsqueeze(1), hidden)

        dec_state = torch.cat((hidden[0].view(-1, config.hidden_dim), hidden[1].view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        sec_attn_dist = self.attention_section_network(dec_state, encoder_sec_outputs, sec_padding_mask)
        context, attn_dist, coverage_next = self.attention_network(dec_state, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, sec_attn_dist, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((context, dec_state, inp), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), context), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, hidden, context, attn_dist, p_gen, coverage

class Model(nn.Module):
    def __init__(self, model_file_path=None):
        super(Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        decoder.embedding.weight = encoder.embedding.weight

        self.encoder = encoder
        self.decoder = decoder

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)

    def forward(self, batch):
        enc_batch, enc_padding_mask, sec_padding_mask, enc_lens, enc_sec_lens, enc_batch_extend_vocab, extra_zeros, context, coverage = \
            get_input_from_batch(batch, config.use_gpu)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, config.use_gpu)

        encoder_outputs, encoder_feature, encoder_sec_outputs, hidden = self.encoder(enc_batch, enc_lens, enc_sec_lens, batch.sec_num, batch.sec_len)

        step_losses = []
        preds = []
        for t in range(min(max_dec_len, config.max_dec_steps)):
            inputs = dec_batch[:, t]  # Teacher forcing
            final_dist, hidden,  context, attn_dist, p_gen, next_coverage = self.decoder(inputs, hidden,
                                                        encoder_outputs, encoder_feature, encoder_sec_outputs, enc_padding_mask, sec_padding_mask, context,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, t)
            target = target_batch[:, t].unsqueeze(1)
            preds.append(final_dist[0].argmax().item())
            gold_probs = torch.gather(final_dist, 1, target).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, t]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss, preds
