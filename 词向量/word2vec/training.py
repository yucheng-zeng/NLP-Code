# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def my_function():
    wiki_news = open('../data/reduce_zhiwiki.txt', 'r')
    '''
    sg = 0 表示使用CBOW训练词向量, sg = 1表示利用Skip-gram训练词向量
    size 表示向量的维度
    windows表示当前词和预测词可能的最大距离
    min_count表示最小出现次数, 如果一个词语的出现次数小于min_count,那么直接忽略这个词
    workers 表示训练所使用的线程数
    '''
    model = Word2Vec(LineSentence(wiki_news), sg=0,size=192, window=5, min_count=5, workers=9)
    model.save('./data/zhiwiki_news.word2vec')

if __name__ == '__main__':
    my_function()
