# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
import jieba
from langconv import *

# 处理预料库信息
def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = '../data/zhwiki-latest-pages-articles.xml.bz2'  # 语料库路径
    f = open('../data/reduce_zhiwiki.txt', 'w')  # 处理后的语料库路径
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})  # 加载维基百科语料库
    for text in wiki.get_texts():  # 获取语料库中的每段文本
        for temp_sentence in text:  # 以句子为基本单位读取一段文本
            temp_sentence = Converter('zh-hans').convert(temp_sentence)  # 将繁体字转换为简体字
            seg_list = list(jieba.cut(temp_sentence))  # 分词
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')  # 将每一个文本转化为以词为单位,写进文件之中
        l = []
        i = i + 1

        # 报数
        if (i %200 == 0):
            print('Saved ' + str(i) + ' articles')
    f.close()

if __name__ == '__main__':
    my_function()
