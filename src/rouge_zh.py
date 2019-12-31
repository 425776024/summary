#!/usr/bin/env python
# coding: utf-8


def compute_rouge_n(text1, text2, n):
    # 真实结果text2，给预测文本text1，计算分数
    def ngram(text, n):
        leng = len(text)
        word_dic = {}
        for i in range(0, leng, n):
            start = i
            words = ""
            if leng - start < n:
                break
            else:
                words = text[start: start + n]
                word_dic[words] = 1
        return word_dic

    dic1 = ngram(text1, n)
    dic2 = ngram(text2, n)
    x = 0
    y = len(dic2)
    for w in dic1:
        if w in dic2:
            x += 1
    rouge = x / y
    return rouge if rouge <= 1.0 else 1.0
