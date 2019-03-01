# -*- coding: utf-8 -*-
import codecs
import numpy
import gensim
import numpy as np
from keyword_extract import *

wordvec_size=192


'''
计算两篇文章的相似度
'''
def get_char_pos(string,char):
    '''
    找出string中char的索引
    :param string: 待查找字符串
    :param char: 目标字符
    :return: 索引列表
    '''
    chPos=[]
    try:
        chPos=list(((pos) for pos,val in enumerate(string) if(val == char)))
    except:
        pass
    return chPos

def word2vec(file_name, model_):
    """
    计算file_name中的词的词向量。
    如果在model_中能找到就设置为model_中相应的词向量，否则设置为默认的全0向量
    :param file_name:
    :param model_:
    :return:
    """
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        word_vec_all = np.zeros(wordvec_size)
        for data in f:
            # 通过提前设置一个假的空格索引 -1 来处理第一个词的索引
            space_pos = [-1]
            space_pos.extend(get_char_pos(data, ' '))  # 将空格的的索引值都标记出来

            # 书上的代码有错。
            # 已改成从space_pos[i]+1开始切片，否则会包含空格，
            # 使得在model中查找不到这个词，导致相似度偏低
            for i in range(len(space_pos)-1):
                word = data[space_pos[i]+1: space_pos[i+1]]  # 获取这个词

                if model_.__contains__(word):  # HUOQU取这个词的词向量
                    # print("No.%d word: %s" % (i, word))
                    word_vec_all += model_[word]  # 将这个词的词向量加到文章关键词的此项量

    return word_vec_all

# 计算向量一与向量二的余弦相似度
def simlarityCalu(vector1,vector2):
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('./data/zhiwiki_news.word2vec')
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywords.txt'
    p2_keywords = './data/P2_keywords.txt'
    getKeywords(p1, p1_keywords)
    getKeywords(p2, p2_keywords)
    p1_vec=word2vec(p1_keywords,model)
    p2_vec=word2vec(p2_keywords,model)

    print(simlarityCalu(p1_vec,p2_vec))
