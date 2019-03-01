# 给一个词语上标签


class word2train(object):
    def __init__(self):
        self.inputpath = ''
        self.trainoutputpath =''
        self.testoutputpath = ''

    def makeLabel(self,text):
        out_text = []
        # 长度为1,为单独成词
        if len(text) == 1:
            out_text.append('S')
        # 长度不为一,则为B[M]E形式
        else:
            out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
        return out_text

    def split_data(self,path1,path2,path3):
        self.inputpath = path1
        self.trainoutputpath = path2
        self.testoutputpath = path3
        index = 0
        with open(self.inputpath,'r') as inputobject,\
                open(self.trainoutputpath,'w') as train_outputobject,\
                open(self.testoutputpath,'w') as test_outputobject:
                    for line in inputobject.readlines():
                        if len(line)==0:
                            continue
                        line = line.split()
                        is_test = True if index % 8 == 0 else False
                        index += 1
                        save_obj = test_outputobject if is_test else train_outputobject  # 选择存储目标
                        for word in line:
                            if len(word)==0:
                                continue
                            taglist = self.makeLabel(word)
                            for k, v in enumerate(word):
                                save_obj.write(v+'\t')
                                save_obj.write(taglist[k]+'\n')

    def load_model(self,path):
        import os, CRFPP
        if os.path.exists(path):
            return CRFPP.Tagger('-m{0} -v -3 -n2'.format(path))
        return None

    def test(self,path):
        with open(path,'r') as fileobject:
            all_tag = 0  # 记录所有的标记数
            tag = 0  # 记录真实的标记数
            pred_tag = 0  # 记录预测的标记数
            correct_tag = 0  # 记录正确的标记数
            correct_split_tag = 0  # 记录正确的词标记数

            # r 为真实值， p 为预测值
            states = ['B', 'M', 'E', 'S']
            for line in fileobject.readlines():
                line = line.strip()
                if line == '':
                    continue
                _, r, p = line.split()

                all_tag += 1

                if r == p:
                    correct_tag += 1
                    if r in states:
                        correct_split_tag += 1
                if r in states:
                    tag += 1
                if p in states:
                    pred_tag += 1

            precious = 1.0 * correct_tag / pred_tag
            recall = 1.0 * correct_tag / tag
            print(all_tag,correct_tag,correct_split_tag,tag,pred_tag)
            print('loc_P:{0}, loc_R:{1}, loc_F1:{2}'.format(precious, recall, (2 * precious * recall) / (recall+ precious)))

    def predict(self,text):
        tagger = self.load_model('./data/model')
        for c in text:  # 遍历测试语句
            tagger.add(c)  # 将语句加载到模板当中去

        result = []  # 记录结果
        # parse and change internal stated as 'parsed'
        tagger.parse()
        word = ''
        # size为语句词个数, xsize为每个词由多少个字组成
        for i in range(0, tagger.size()):
            for j in range(0, tagger.xsize()):
                ch = tagger.x(i, j)  # 获取词
                tag = tagger.y2(i)  # 获取词标签
                if tag == 'B':
                    word = ch
                elif tag == 'M':
                    word += ch
                elif tag == 'E':
                    word += ch
                    result.append(word)
                elif tag == 'S':
                    word = ch
                    result.append(word)
        return result


if __name__=='__main__':
    #train = word2train('./data/trainCorpus.txt_utf8','./data/train.txt','./data/test.txt')
    #train.start()
    #train.test('./data/test.rst')
    train = word2train()
    text = '当代大学生应该关注精神文明生活'
    result = train.predict(text)
    print(result)
