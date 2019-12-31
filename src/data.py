import os, sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

import glob
import random
import struct
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


# 不出现在词汇表中：<s>, </s>, [PAD], [UNK], [START], [STOP]


class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    break

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count


def text_generator(data_path):
    while True:
        # 文件路径集合
        filelist = glob.glob(data_path)
        assert filelist, ('Error: Empty filelist at %s' % data_path)
        random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            temp_tuple_txt = []
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                e = example_pb2.Example.FromString(example_str)
                article = e.features.feature['article'].bytes_list.value[0]
                abstract = e.features.feature['abstract'].bytes_list.value[0]
                article, abstract = article.decode(), abstract.decode()
                temp_tuple_txt.append((article, abstract))
            for t in temp_tuple_txt:
                yield t
        print("example_generator done")


def article2ids(article_words, vocab):
    # 把文章变成ids，同时返回其oov词汇表
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
                # 在OOV词汇表中的位置
                oov_num = oovs.index(w)
                # OOV词的最终ID是词汇表长+OOV词汇表相对长位置ID
                ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            # 如果是OOV单词，用OOV单词的ID表示
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:  # 否则，还是UNK的ID
                ids.append(unk_id)
        else:  # 否则就是词汇表的词
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    '''
    :param id_list: 输入序号列，序号可能超出vocab大小，就往article_oovs去拿
    :param vocab:
    :param article_oovs:
    :return: 返回真正的词
    '''
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # 可能是 [UNK]词
        except ValueError as e:  # w 是 OOV词
            assert article_oovs is not None, "错误，需要有OOV词汇表"
            # 拿到OOV词汇表的索引
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:
                raise ValueError('序号超出OOV词汇表范围')
        words.append(w)
    return words


def abstract2sents(abstract):
    """abstract 是用<s></s>分割的多句话，需要分割成n句话的列表并去掉句子分割符号"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:
                # pointer-generator mode
                pass
            if w in article_oovs:
                new_words.append("__%s__" % w)
            else:
                new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
