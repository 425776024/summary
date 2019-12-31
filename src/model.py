import os, sys

# 添加project目录至环境变量
base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
sys.path.append(base_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from config import USE_CUDA, DEVICE


# 权重初始化，默认xavier
def init_network(model, method='xavier', seed=123):
    for name, w in model.named_parameters():
        # if exclude not in name:
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    def forward(self, input_x, seq_lens):
        # input_x:[batch,seq_lens]
        # embedded:[batch,seq_lens,emb_dim]
        embedded = self.embedding(input_x)
        # 压紧，将填充过的变长序列压紧
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        # [batch, seq_lens, 2*hid_dim]
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)

        # [batch, max(seq_lens), 2*hid_dim]
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，
        # 这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        # 判断是否contiguous用torch.Tensor.is_contiguous()函数。
        encoder_outputs = encoder_outputs.contiguous()

        # [batch*max(seq_lens), 2*hid_dim]
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)

        # hidden：2 * [2, batch, hid_dim])
        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def forward(self, hidden):
        # h, c dim = [2, batch, hidden_dim]
        h, c = hidden
        # [batch, hidden_dim*2]
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        # [batch, hidden_dim]
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        # h, c dim = [1, batch, hidden_dim]
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        # s_t_hat：decoder 输出的每个state的hidden，[B , 2*hid_dim]
        dec_fea = self.decode_proj(s_t_hat)
        # B x t_k x 2*hid_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        # [B * t_k , 2*hid_dim]
        dec_fea_expanded = dec_fea_expanded.view(-1, n)

        # [B * t_k , 2*hidden_dim]，相当于拿decoder输出state和encoder out融合，生成attention特征
        att_features = encoder_feature + dec_fea_expanded
        if config.is_coverage:
            # [B * t_k , 1]，覆盖向量
            coverage_input = coverage.view(-1, 1)
            # B * t_k x 2*hidden_dim
            coverage_feature = self.W_c(coverage_input)
            # 和覆盖向量融合，计算最终attenion值
            att_features = att_features + coverage_feature

        # B * t_k x 2*hidden_dim
        e = torch.tanh(att_features)
        # B * t_k x 1
        scores = self.v(e)
        # B x t_k
        scores = scores.view(-1, t_k)

        # [B , t_k],attention weight
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask
        # 因为乘了enc_padding_mask，所以不一定和为1了
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        # [B , 1 , t_k]
        attn_dist = attn_dist.unsqueeze(1)
        # [B , 1 , 2*hidden_dim]
        c_t = torch.bmm(attn_dist, encoder_outputs)
        # [B , 2*hidden_dim]，每个batch的attention上下文向量
        c_t = c_t.view(-1, config.hidden_dim * 2)

        # [B , t_k]
        attn_dist = attn_dist.view(-1, t_k)

        if config.is_coverage:
            # [B , t_k]，积累到当前时刻的attn_dist和，覆盖向量
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        # y_t_1摘要的一个单词，batch里的每个句子的同一位置的单词编码
        # s_t_1输入的hidden state
        # c_t_1输入之前的上下文向量，batch_size, 2 * config.hidden_dim

        y_t_1_embd = self.embedding(y_t_1)
        # teacher forcing训练词向量和上下文向量合并，再从新映射为emb_dim，
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        # 进行单向lstm解码，输入包含了训练词和读取encoder out全文的上下文
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        # s_t:2*[1,B,hidden_dim]
        h_decoder, c_decoder = s_t
        # [B , 2*hidden_dim]
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)
        # 本轮的attention上下文向量
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            # 计算p_gen数
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        # B x hidden_dim * 3
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)
        output = self.out1(output)  # B x hidden_dim
        output = self.out2(output)  # B x vocab_size
        # 词汇表的预测分布
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            # 新的预测词概率分布计算
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist # [B , t_k]

            if extra_zeros is not None:
                # 原来词汇表分布，和oov词汇表分布，拼接！extra_zeros：[batch_size, batch.max_art_oovs]
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            # 还只是形状上有词汇表，和oov表分布，还需要填充来oov表分布的值，变成最终预测词概率分布
            # scatter_add：将attn_dist_中的数据，按照enc_batch_extend_vocab中的索引位置，添加至vocab_dist_矩阵中
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()
        # 参数初始化
        for model in [encoder, decoder, reduce_state]:
            init_network(model)

        # decoder与encoder参数共享
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if USE_CUDA:
            encoder = encoder.to(DEVICE)
            decoder = decoder.to(DEVICE)
            reduce_state = reduce_state.to(DEVICE)

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
