import os, sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

import random
import torch
import numpy as np
from random import shuffle
from queue import Queue
from torch.autograd import Variable

import data
import config
from config import USE_CUDA, DEVICE

random.seed(1234)


class Example(object):

    def __init__(self, article, abstract, vocab):
        # 开始、结束 ID
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)
        # 编码 article，不在vocab的用UNK序号表示，拿这个做输入
        self.enc_input = [vocab.word2id(w) for w in article_words]
        # 处理 abstract，不在vocab的用UNK序号表示
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # 解码阶段的输入序列和输出序列
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps,
                                                                 start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # 如果使用pointer-generator模式,enc_input中unk对应的0序号，会替换为词汇表长+oov词汇表内位置的序号
        if config.pointer_gen:
            # 编码输入扩展了oov词的序列（unk的有了序号）和oov单词
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # 获取参考摘要的id，其中oov单词由原文中的oov单词编码表示
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # 新的目标序列，unk词有序号
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # 存储原始数据
        self.original_article = article
        self.original_abstract = abstract
        # 编码前的摘要，单词列表
        self.original_abstract_sents = abstract

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # 截断
            inp = inp[:max_len]
            target = target[:max_len]  # 没有结束标志
        else:  # 无截断
            target.append(stop_id)  # 结束标志
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        # example_list：Expamle对象列表
        self.batch_size = batch_size
        self.vocab = vocab
        # 填充的ID
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # example_list中每一个Example对象，encoder长度填充到max_enc_seq_len
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # 非填充部分为1
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100

    def __init__(self, data_path, vocab, batch_size):
        self._data_path = data_path
        self._vocab = vocab
        self.batch_size = batch_size
        #
        self.text_gen = data.text_generator(self._data_path)
        self.batch_gen = self.batch_generator()
        self.example_gen = self.example_generator()

    def next_batch(self):
        batch = self.batch_gen.__next__()
        return batch

    def example_generator(self):
        # 文章、摘要生成器
        while True:
            try:
                (article, abstract) = self.text_gen.__next__()
            except StopIteration:
                print("example generator 迭代结束")
                break
            # 编码abstract
            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
            # 处理成一个Example.
            example = Example(article, abstract_sentences[0], self._vocab)
            # 放Example对象到队列
            yield example

    def batch_generator(self):
        # yield a Batch
        while True:
            try:
                inputs = []
                for _ in range(self.batch_size):
                    ex = self.example_gen.__next__()
                    inputs.append(ex)
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)
                batch = Batch(inputs, self._vocab, self.batch_size)
                yield batch
            except StopIteration:
                print("batch_generator 迭代结束")
                break


def get_input_from_batch(batch):
    # 解析Batch对象为模型的输入数据
    batch_size = len(batch.enc_lens)

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if USE_CUDA:
        enc_batch = enc_batch.to(DEVICE)
        enc_padding_mask = enc_padding_mask.to(DEVICE)

    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(DEVICE)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(DEVICE)
    c_t_1 = c_t_1.to(DEVICE)

    if coverage is not None:
        coverage = coverage.to(DEVICE)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


def get_output_from_batch(batch):
    # 解析batch为解码阶段，和目标序列的数据
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if USE_CUDA:
        dec_batch = dec_batch.to(DEVICE)
        dec_padding_mask = dec_padding_mask.to(DEVICE)
        dec_lens_var = dec_lens_var.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
