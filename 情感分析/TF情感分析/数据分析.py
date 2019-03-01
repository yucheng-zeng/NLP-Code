import os
from os.path import isfile, join
import matplotlib
from matplotlib import pyplot
'''
计算文本的而平均长度
'''


def analyse_allfile():
    num_words = []
    analyse_posfile(num_words)
    analyse_negfile(num_words)
    num_files = len(num_words)
    print('文件总数', num_files)
    print('所有的词的数量', sum(num_words))
    print('平均文件词的长度', sum(num_words) / len(num_words))
    visual(num_words)

def analyse_posfile(num_words):
    pos_files = ['../pos/' + f for f in os.listdir('../pos/') if isfile(join('../pos/', f))]
    for pf in pos_files:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('正面评价完结')

def analyse_negfile(num_words):
    neg_files = ['../neg/' + f for f in os.listdir('../neg/') if isfile(join('../neg/', f))]
    for nf in neg_files:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('负面评价完结')

def visual(num_words):
    matplotlib.rcParams['font.sans-serif'] = ['Simhei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    pyplot.hist(num_words,50, facecolor='r')
    pyplot.xlabel('文本长度')
    pyplot.ylabel('频次')
    pyplot.axis([0,1200,0,8000])
    pyplot.show()

if __name__=='__main__':
    analyse_allfile()