import jieba
jieba.load_userdict('user_dict.txt')
import operator

'''
窗口假设效果一般
'''


class TextRank(object):
    def __init__(self,path):
        self.filename = path

    def start(self, iter = 100, d = 0.5, window_length=5):
        '''
        :param iter: 最大迭代次数
        :param d: 权值d
        :param window_length:窗口长度
        :return:
        '''
        context = self.read_file(self.filename)
        all_wordlist = []
        # 将文本变为词列表
        for sentence in context:
            if sentence:
                all_wordlist.extend(self.cut(sentence))

        stop_word = self.load_stopword('stop_word.txt')
        WS_dict = {}
        # 初始化并且去掉停用词
        wordlist =  []
        for word in all_wordlist:
            if word in stop_word:
                continue
            else:
                WS_dict[word] = 1.1
                wordlist.append(word)

        diff1 = 0  # 记录词语的总体权值差异
        length = len(wordlist)  # 文本总词数
        for i in range(0,iter):
            # 定义窗口位置
            first = 0  # 窗口开始位置
            last = window_length  # 窗口结束位置
            while last <= length:
                windows = wordlist[first:last]  # 获取窗口
                for index in range(0,last-first):
                    current_index = index
                    weight = 0
                    for other_index in range(0,last-first):
                        if other_index==current_index:
                            continue
                        weight += (1.0/(window_length-1))*WS_dict[windows[other_index]]
                    temp1 = (1.0-d) + d*weight
                    temp2 = WS_dict[windows[current_index]]
                    diff1 += abs(temp2 - temp1)
                    WS_dict[windows[current_index]] = (1.0-d) + d*weight
                first += 1
                last += 1

            # 判断是否是收敛
            if i == 0:
                diff2 = diff1
                continue
            if diff2-diff1<1e-5:
                break
            diff2 = diff1

        WS_dict = sorted(WS_dict.items(),key=operator.itemgetter(1),reverse=True)  # 返回前N个权重最大的词语
        print(WS_dict)

    # 读取文件内容
    def read_file(self, path):
        context = []
        with open(path, 'r', encoding='UTF-8-sig') as fileobject:
            for line in fileobject.readlines():
                # 利用正则表达式去掉一些一些标点符号之类的符号。
                line = line.strip()
                context.append(line)
        return context

    # 加载停用词
    def load_stopword(self,filename):
        stop_word = []
        with open(filename, 'r') as fileobject:
            for line in fileobject.readlines():
                stop_word.append(line.strip())
        return stop_word

    # 切分句子变为词语
    def cut(self, text):
        return list(jieba.cut(text, cut_all=False))

if __name__=='__main__':
    rank = TextRank('./test.txt')
    rank.start()