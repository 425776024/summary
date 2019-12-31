import os
import sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

import time
import argparse

import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import config
from data import Vocab
from config import USE_CUDA, DEVICE
from batcher import Batcher
from batcher import get_input_from_batch
from batcher import get_output_from_batch


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, batch_size=config.batch_size)
        train_dir = os.path.join(config.log_root)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, loss, iter_step, name=None):
        state = {
            'iter': iter_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': loss
        }
        if name is None:
            name = 'model_{}_{}'.format(iter_step, loss)
        model_save_path = os.path.join(self.model_dir, name)
        torch.save(state, model_save_path)
        print('saved loss:', loss)

    def setup_train(self, model_file_path=None):
        # 初始化模型
        self.model = Model(model_file_path)
        # 模型参数的列表
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        # 定义优化器
        self.optimizer = optim.Adam(params, lr=config.adam_lr)
        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0
        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(DEVICE)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        # enc_batch是包含unk的序列
        # c_t_1是初始上下文向量
        # extra_zeros:oov词汇表概率，[batch_size, batch.max_art_oovs]
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        # dec_batch是普通摘要序列，包含unk，target_batch是目标词序列，不包含unk，unk的词用len(vocabe)+oov相对位置代替
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)
        self.optimizer.zero_grad()

        # [batch, seq_lens, 2*hid_dim]，[batch*max(seq_lens), 2*hid_dim]，[2, batch, hid_dim])
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        # (h,c) = ([1, batch, hid_dim], [1, batch, hid_dim])
        # 之前的hidden state是双向的[2, batch, hid_dim]，需要转成1维的[1, batch, hid_dim]，作为新的decoder的hidden输入
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            y_t_1 = dec_batch[:, di]
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的vocab size
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            # 摘要的下一个单词的编码，[B]
            target = target_batch[:, di]
            # [B,1]
            target_i = target.unsqueeze(1)
            # 取出目标单词的概率//取出final_dist中，target中对应位置的数据（对于目标单词预测的概率）
            gold_probs = torch.gather(final_dist, 1, target_i).squeeze()

            # 单个词的预测损失
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                # 取当前t步attention向量，和之前t-1步attention和向量，的min值做sum，当作额外的coverage loss来压制重复生成。
                # 迫使loss让当前第t步的attention向量attn_dist值，尽可能比之前t-1步attention和向量的值小。（大的的attention值意味着之前可能被预测生成了这个词）
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                # 加个\lambda 系数，表示多大程度考虑这个压制重复的coverage loss
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                # 初始时的coverage覆盖向量，就更新成累加了
                coverage = next_coverage
            # mask的部位不计入损失
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # 序列的整体损失
        batch_avg_loss = sum_losses / dec_lens_var
        # 整个batch的整体损失
        loss = torch.mean(batch_avg_loss)

        loss.backward()

#         self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
#         clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
#         clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, model_file_path=None):
        # 训练设置，包括
        iter_data, loss = self.setup_train(model_file_path)
        start = time.time()
        # 总数据量data_size，轮回训练iter_loop次数据
        data_size = 80000
        i = 0
        min_loss = 10000
        while iter_data < data_size * config.iter_loop:
            # 获取下一个batch数据
            batch = self.batcher.next_batch()
            iter_data += batch.batch_size
            loss = self.train_one_batch(batch)
            i += 1
            if i % 100 == 0:
                print('steps %d, seconds for %d ,' % (i, time.time() - start))
                print('loss:', loss)
                start = time.time()
            if loss < min_loss and i > 1000:
                min_loss = loss
                self.save_model(loss, i, name='best_loss')


def init_print():
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("时间:{}".format(stamp))
    print("***参数:***")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            print(":".join([k, str(v)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    init_print()
    train_processor = Train()
    train_processor.trainIters(args.model_path)
