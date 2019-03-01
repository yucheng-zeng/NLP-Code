from numpy import *

def loadDataSet(pos_path,neg_path):
    # 创建特征集
    wordList = []
    classVec = []
    with open(pos_path,'r') as fileobject:
        for line in fileobject.readlines():
            line = line.split()
            wordList.append(line)
            classVec.append(1)
    with open(neg_path, 'r') as fileobject:
        for line in fileobject.readlines():
            line = line.split()
            wordList.append(line)
            classVec.append(0)
    return wordList, classVec

# 创建词列表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建数据集合
    for document in dataSet:  # 遍历数据集
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)   # 一个列表

# 计算p(wi|c1) p(wi|c0) p(c1) p(c0)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  # 获取训练样本的个数
    numWords = len(trainMatrix[0])  # 获取每个样本向量的维度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算标签为1的样本的先验概率
    # 这样初始化的作用为避免出现p(wi|ci)为0的情况
    p0Num = ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p1Num = ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p0Denom = 2.0  # 初始化分母
    p1Denom = 2.0  # 初始化分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果该样本的标签为1, 既是侮辱性文本
            p1Num += trainMatrix[i]  # 两个向量相加
            p1Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
        else:  # 如果该样本的标签为0, 既是非侮辱性文本
            p0Num += trainMatrix[i]  # 两个向量相加
            p0Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
    p1Vect = log(p1Num/p1Denom)  # 计算p(wi|c1)  取对数是防止下溢出
    p0Vect = log(p0Num/p0Denom)  # 计算p(wi|c0)
    return p0Vect, p1Vect, pAbusive



# 创建词向量 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 每个词可以出现多次
    return returnVec

# 切分文本
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 获取所有单词, 并且去除空格和标点符号, 将所有单词变为小写


if __name__ == '__main__':
    wordList, classVec = loadDataSet('./data/pos_file.txt','./data/neg_file/txt')
    for key, value in enumerate(classVec):
        print(value,wordList[key])
    print(len(wordList),len(classVec))