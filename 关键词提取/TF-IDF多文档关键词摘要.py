import jieba
jieba.load_userdict('user_dict.txt')


class TF_IDF(object):

    def __init__(self,path):
        self.folder = path
        self.filename_list = []
        self.word_dict = {}

    # 获取全部文档
    def listdir(self,path):  # 传入存储的list
        import os
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path)
            else:
                self.filename_list.append(file_path)
    # 获取IDF
    def read_all_word(self):
        self.listdir(self.folder)
        for file in self.filename_list:
            loaded_word = []
            context = self.read_file(file)
            for sentence in context:
                if not sentence:
                    continue
                wordlist = set(self.cut(sentence))
                for word in wordlist:
                    if word not in loaded_word:
                        self.word_dict[word] = self.word_dict.get(word,0) + 1
                        loaded_word.extend(word)
        for key, value in self.word_dict.items():
            print(key,value)

    # 计算TF-IDF 并且获取关键词
    def find_keyword(self):
        import math
        import operator
        self.listdir(self.folder)
        self.read_all_word()
        D = len(self.filename_list)
        # 计算TF-IDF
        stop_word = self.load_stopword('stop_word.txt')
        for file in self.filename_list:
            current_dict = {}
            context = self.read_file(file)
            all_count = 0
            current_keyword = dict()
            for sentence in context:
                if not sentence:
                    continue
                wordlist = set(self.cut(sentence))
                # 去除停用词
                for word in wordlist:
                    if word in stop_word:
                        continue
                    current_dict[word] = current_dict.get(word,0) + 1
                    all_count += 1
            # 计算TF-IDF
            for key_word, value in current_dict.items():
                if self.is_illegal(key_word):
                    continue
                # +1平滑
                if key_word in self.word_dict.keys():
                    Di = self.word_dict[key_word] + 1
                else:
                    Di = 1
                count = value
                # 计算idf
                idf = math.log(1.0*D/Di,2)
                # 计算tf
                tf = 1.0*count/all_count
                # 计算tf-idf
                current_keyword[key_word] = tf*idf
            # 返回权重最高的前N个词
            current_top_keyword = sorted(current_keyword.items(),key=operator.itemgetter(1),reverse=True)[:5]
            print(current_top_keyword)
            # 协会到文件之中
            self.write2file('result.txt',file,current_top_keyword)

    # 加载停用词表
    def load_stopword(self,filename):
        stop_word = []
        with open(filename, 'r') as fileobject:
            for line in fileobject.readlines():
                stop_word.append(line.strip())
        return stop_word

    # 将文件之中的内容读取出来
    def read_file(self,path):
        context = []
        with open(path,'r',encoding='UTF-8-sig') as fileobject:
            for line in fileobject.readlines():
                # 利用正则表达式去掉一些一些标点符号之类的符号。
                line = line.strip()
                context.append(line)
        return context

    # 分词
    def cut(self,text):
        return list(jieba.cut(text,cut_all=True))

    # 判断是否是合法支付
    def is_illegal(self,text):
        return text.isdigit()

    # 将结果写回到文件之中
    def write2file(self,filename,file, targetdict):
        with open(filename,'a') as fileobejct:
            num = file.split('/')[-1]
            fileobejct.write(num+'\t')
            for key_value in targetdict:
                fileobejct.write(key_value[0]+' ')
                fileobejct.write(str(key_value[1])+'\t')
            fileobejct.write('\n')

if __name__=='__main__':
    tf_idf = TF_IDF('./news/金融新闻')
    tf_idf.find_keyword()