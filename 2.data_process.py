import os
import sys
import time
import jieba


def load_data(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = jieba.cut(line.strip())
            word_list = list(words)
            data_list.append(' '.join(word_list).strip())
    return data_list


def build_train_val(article_data, summary_data, train_num=600000):
    '''
    合并文章、摘要
    :param article_data:
    :param summary_data:
    :param train_num:
    :return:
    '''
    train_list = []
    val_list = []
    n = 0
    for text, summ in zip(article_data, summary_data):
        n += 1
        if n <= train_num:
            train_list.append(text)
            train_list.append(summ)
        else:
            val_list.append(text)
            val_list.append(summ)
    return train_list, val_list


def save_file(filename, li):
    with open(filename, 'w+', encoding='utf-8') as f:
        for item in li:
            f.write(item + '\n')
    print("Save OK")


data_path = 'data/'
# 加载原来的文本、摘要数据
ARTICLE_FILE = data_path + "train_text.txt"
SUMMARRY_FILE = data_path + "train_label.txt"

# 保存训练、测试集数据
TRAIN_FILE = data_path + "train_art_summ_prep.txt"
VAL_FILE = data_path + "val_art_summ_prep.txt"
# 加载用户字典
user_dict = data_path + 'user_dict.txt'
jieba.load_userdict(user_dict)

article_data = load_data(ARTICLE_FILE)
summary_data = load_data(SUMMARRY_FILE)

# 多少划分去训练，剩下的去预测
TRAIN_SPLIT = 80000
# 分成训练、校验数据
train_list, val_list = build_train_val(article_data, summary_data, train_num=TRAIN_SPLIT)
save_file(TRAIN_FILE, train_list)
save_file(VAL_FILE, val_list)
