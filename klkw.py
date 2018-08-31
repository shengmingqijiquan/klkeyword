# -*- coding: utf-8 -*-
import codecs
import csv
import math
import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import numpy as np
import pandas as pd

def data_write_csv(keyword):#file_name为写入CSV文件的路径，datas为要写入数据列表
    output = open('output.csv', 'a', newline='',encoding='gbk')
    csv_write = csv.writer(output, dialect='excel')
    csv_write.writerow(keyword)
    print("保存文件成功，处理结束")

# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,errors='ignore').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list

if __name__ == '__main__':
    #text = open('knowledge.txt', encoding="utf-8",errors='ignore').read()
    with open('knowledge.txt',encoding='UTF-8') as f:
         line = f.readline()
         print('TF-IDF模型结果：')
         pos = True
         while line:
             seg_list = seg_to_list(line, pos)
             filter_list = word_filter(seg_list, pos)
             tfidf_model = TfidfVectorizer()
             tfidf_matrix = tfidf_model.fit_transform(filter_list)
             word_dict = tfidf_model.get_feature_names()
             print(word_dict)
             data_write_csv(word_dict)
             line = f.readline()



