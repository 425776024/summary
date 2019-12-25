import os
import struct
import collections
from tensorflow.core.example import example_pb2


def chunk_file(finished_files_dir, chunks_dir, name, chunk_size):
    """构建二进制文件"""
    in_file = os.path.join(finished_files_dir, '%s.bin' % name)
    print(in_file)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (name, chunk))  # 新的分块
        with open(chunk_fname, 'wb') as writer:
            for _ in range(chunk_size):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # 创建一个文件夹来保存分块
    if not os.path.isdir(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)
    # 将数据分块
    for name in ['train', 'val']:
        print("Splitting %s data into chunks..." % name)
        chunk_file(FINISHED_FILE_DIR, CHUNKS_DIR, name, CHUNK_SIZE)
    print("Saved chunked data in %s" % CHUNKS_DIR)


def read_text_file(text_file):
    """从预处理好的文件中加载数据"""
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(input_file, out_file, makevocab=False):
    """生成模型需要的文件"""
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        # 读取输入的文本文件，使偶数行成为article，奇数行成为abstract（行号从0开始）
        lines = read_text_file(input_file)
        for i, new_line in enumerate(lines):
            if i % 2 == 0:
                article = lines[i]
            if i % 2 != 0:
                abstract = "%s %s %s" % (SENTENCE_START, lines[i], SENTENCE_END)

                # 写入tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding='utf-8')])
                tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, encoding='utf-8')])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # 如果可以，将词典写入文件
                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START, SENTENCE_END]]  # 从词典中删除这些符号
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens]  # 去掉句子开头结尾的空字符
                    tokens = [t for t in tokens if t != ""]  # 删除空行
                    vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # 将词典写入文件
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(FINISHED_FILE_DIR, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


FINISHED_FILE_DIR='data/'
TRAIN_FILE = FINISHED_FILE_DIR+"train_art_summ_prep.txt"
VAL_FILE = FINISHED_FILE_DIR+"val_art_summ_prep.txt"


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 50000  # 词汇表大小
CHUNK_SIZE = 1000  # 每个分块example的数量，用于分块的数据

# tf模型数据文件存放目录
CHUNKS_DIR = os.path.join(FINISHED_FILE_DIR, 'chunked')


if not os.path.exists(FINISHED_FILE_DIR):
    os.makedirs(FINISHED_FILE_DIR)
    
write_to_bin(VAL_FILE, os.path.join(FINISHED_FILE_DIR, "val.bin"))
write_to_bin(TRAIN_FILE, os.path.join(FINISHED_FILE_DIR, "train.bin"), makevocab=True)
chunk_all()
