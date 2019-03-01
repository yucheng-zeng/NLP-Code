import jieba

if __name__=='__main__':
    sentence = '中文分词是文本处理不可或缺的一步'

    # 全模式
    seg_list = list(jieba. cut(sentence,cut_all=True))
    print('全模式：',seg_list)

    #精确模式
    seg_list = list(jieba.cut(sentence, cut_all=False))
    print('精确模式：', seg_list)

    #默认精确模式
    seg_list = list(jieba.cut(sentence, cut_all=True))
    print('默认精确模式：', seg_list)

    #搜索引擎模式
    seg_list = list(jieba.cut_for_search(sentence))
    print('搜索引擎模式：', seg_list)
