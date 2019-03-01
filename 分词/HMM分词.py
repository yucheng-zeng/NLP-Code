
class HMM(object):
    def __init__(self,filename='./data/hmm_model.pkl'):
        import os
        # 主要是用于存取算法中间结果，不用每次都训练模型
        self.model_file = filename
        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        # 参数加载，看是否要重新训练模型
        self.load_para = False

    # 用于加载已计算的中间结果，当需要重新训练时，需初始化清空结果
    def try_load_model(self, trained):
        if trained:
            # 读取参数
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率（状态->状态的条件概率, 在知道前一个状态下转移到另一个状态的条件概率）
            self.A_dic = {}
            # 发射概率（状态->词语的条件概率, 一个字被标记为相应标记的的条件概率）
            self.B_dic = {}
            # 状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    def split_data(self,path1,path2,paht3):
        with open(path1,'r') as source,\
                open(path2,'w') as train,\
                open(paht3,'w') as test:
            line_num = 0
            for line in source.readlines():
                is_test = True if line_num % 8 == 0 else False
                line_num += 1
                save_obj = test if is_test else train  # 选择存储目标
                save_obj.write(line)

    # 给一个词语上标签
    def makeLabel(self,text):
        out_text = []
        # 长度为1,为单独成词
        if len(text) == 1:
            out_text.append('S')
        # 长度不为一,则为B[M]E形式
        else:
            out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
        return out_text

    # 采用人民日报的分词语料，通过统计，得到HMM所需的初始概率、转移概率以及发射概率
    def train(self, path):
        self.try_load_model(False)
        Count_dic = {}  # 求p(o),每一个标签出现的概率

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list} # 一个M*M矩阵
                self.Pi_dic[state] = 0.0
                self.B_dic[state] = {}
                Count_dic[state] = 0

        init_parameters()
        line_num = -1 # 标记文本句子行数
        # 观察者集合，主要是单个字以及标点等
        words = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                word_list = [i for i in line if i != ' ']  # 获取每一个字
                words |= set(word_list)  # 更新字的集合
                linelist = line.split()  # 将一行切分为词
                line_state = []  # 记录词的标签
                for w in linelist:  # 遍历每行中每一个词
                    line_state.extend(self.makeLabel(w))  # 获取取每一个词的标签
                # print(word_list)
                # print(line_state)
                assert len(word_list) == len(line_state)  # 检查字数是否等于标签数
                for k, v in enumerate(line_state):  # k 记录索引下标, v记录值(标签)
                    Count_dic[v] += 1
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率
                    else:
                        # line_state[k - 1]为上一个标标签
                        self.A_dic[line_state[k - 1]][v] += 1  # 计算转移概率分子
                        # 计算发射概率, p(word|tag)
                        self.B_dic[line_state[k]][word_list[k]] = self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0

        # 转换为概率
        # 每一行的第一个字的标记数目除以行数
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        # 这列k,k1为标签,v前一个为标签k的标记字典, v1为数目, Count_dic[k]标签k出现数目
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}
        #print(self.A_dic)
        # 拉普拉斯 加1平滑 , k为标签,v为标记为标签k的值
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dic.items()}  # 序列化
        #print(self.B_dic)
        # 将模板保存起来
        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)
        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        '''
        :param text: 需要切分的文本
        :param states: 状态序列
        :param start_p: 初始状态
        :param trans_p: 状态转移矩阵
        :param emit_p: 观测概率矩阵
        :return:
        '''

        def IsneverSeen(word, emit_p):
            boolean =   word not in emit_p['S'].keys() and \
                        word not in emit_p['M'].keys() and \
                        word not in emit_p['E'].keys() and \
                        word not in emit_p['B'].keys()
            return boolean

        theta = [{}]  # 记录节点值
        path = {}  # 记录路径值
        # 初始化
        neverSeen = IsneverSeen(text[0], emit_p)
        for y in states:
            if neverSeen:
                theta[0][y] = start_p[y]*1.0
            else:
                theta[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            theta.append({})
            newpath = {}
            #print(text[t])
            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = IsneverSeen(text[t],emit_p)
            for y in states:  # 遍历每一个标签
                # 计算P(word|tag)
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词
                # 计算从上一个节点到当前节点最大概率的路径,加上小数是平滑需要
                (prob, state) = max([((theta[t - 1][y0]+1e-200) * trans_p[y0].get(y, 0) *emitP, y0) for y0 in states])
                theta[t][y] = prob  # 设置当前字节点的概率
                newpath[y] = path[state] + [y]  # 更新路径
            path = newpath

        # 如果最后最后一个词可能不是单独成词, 则最后一个词只能标记为E或者M
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(theta[len(text) - 1][y], y) for y in ('E', 'M')])
        # 否则
        else:
            (prob, state) = max([(theta[len(text) - 1][y], y) for y in states])
        # 返回概率以及路径
        return (prob, path[state])

    def cut(self, text):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        print(prob)
        print(pos_list)
        begin, next = 0, 0
        # 按照路径分词
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]

    def test(self,path):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        all_tag = 0
        pre_tag = 0
        corret_tag = 0
        with open(path,'r') as fileobject:
            for line in fileobject.readlines():
                linelist = line.split()  # 将一行切分为词
                line_state = []  # 记录词的标签
                sentence = ''  # 记录句子
                for w in linelist:  # 遍历每行中每一个词
                    line_state.extend(self.makeLabel(w))  # 获取取每一个词的标签
                    sentence += w
                prob, pos_list = self.viterbi(sentence, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
                for index, pos in enumerate(pos_list):
                    if pos == line_state[index]:
                        corret_tag += 1
                    pre_tag += 1
                    if pos in self.state_list:
                        all_tag += 1
        precious = 1.0*corret_tag/pre_tag
        recall = 1.0*corret_tag/all_tag
        print('Precious:{0}, Recall:{1}, F1:{2}'.format(precious, recall, (2 * precious * recall) / (recall + precious)))


if __name__=='__main__':
    hmm = HMM()
    #hmm.split_data('./data/trainCorpus.txt_utf8','./data/hmmtrain.txt','./data/hmmtest.txt')
    #hmm.train('./data/hmmtrain.txt')
    #hmm.test('./data/hmmtest.txt')
    text = '这是一个非常棒的方案！'
    #text = '当代大学生更应该关注精神方面的健康，因为没有人可以忽视精神方面所带来的问题'
    #text = '我感到肃然悚然而又怅然'
    res = hmm.cut(text)
    print(list(res))
