import numpy as np
import math
import re
import pickle


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


# 创建词向量 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 每个词可以出现多次
    return returnVec


# 创建词列表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建数据集合
    i = 1
    for sentence in dataSet:  # 遍历数据集
        print(i)
        sentence = cleanSentence(sentence[0])
        sentence = sentence.split()
        vocabSet = vocabSet | set(sentence)  # 创建两个集合的并集
        i += 1
    return list(vocabSet)   # 一个列表

# 清理字符串
def cleanSentence(string):
    strip_special_chars = re.compile('[^A-Za-z0-9]+')
    string = string.strip()
    string = string.replace('<br />',' ')
    return re.sub(strip_special_chars,' ',string.lower())

# 朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    :param vec2Classify: 要分类的向量
    :param p0Vec: p(w|c0)
    :param p1Vec: p(w|c1)
    :param pClass1: p(c1)
    :return:
    '''
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)  # 对数相加,既是相乘
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 计算p(wi|c1) p(wi|c0) p(c1) p(c0)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  # 获取训练样本的个数
    print(len(trainMatrix[0]))
    numWords = len(trainMatrix[0])  # 获取每个样本向量的维度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算标签为1的样本的先验概率

    # 这样初始化的作用为避免出现p(wi|ci)为0的情况
    p0Num = np.ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p1Num = np.ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p0Denom = 2.0  # 初始化分母
    p1Denom = 2.0  # 初始化分母
    for i in range(numTrainDocs):
        print(i)
        if trainCategory[i] == 1:  # 如果该样本的标签为1, 既是侮辱性文本
            p1Num += trainMatrix[i]  # 两个向量相加
            p1Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
        else:  # 如果该样本的标签为0, 既是非侮辱性文本
            p0Num += trainMatrix[i]  # 两个向量相加
            p0Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
    print(p1Num)
    print(len(p1Num))
    print(p0Num)
    print(len(p0Num))

    p1Vect = np.zeros(numWords)
    p0Vect = np.zeros(numWords)
    for i in range(numWords):
        p1Vect[i] = math.log(p1Num[i]/p1Denom)
        p0Vect[i] = math.log(p0Num[i]/p0Denom)
    '''
    p1Vect = math.log(p1Num/p1Denom)  # 计算p(wi|c1)  取对数是防止下溢出
    p0Vect = math.log(p0Num/p0Denom)  # 计算p(wi|c0)
    '''
    return p0Vect, p1Vect, pAbusive

# 训练模型
def trainModel():
    listOPosts,listClasses = loadDataSet('./data/pos_file.txt','./data/neg_file.txt')  # 获取数据
    myVocabList = createVocabList(listOPosts)  # 获取词列表
    trainMat=[]  # 初始化训练样本矩阵
    i = 1
    for postinDoc in listOPosts:
        print(i)
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))  # 获取每一个样本对应的词向量
        i += 1
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))  # 计算p(wi|c0) p(wi|c1) p(c1)

    with open('./data/model.pkl','wb') as f:
        pickle.dump(myVocabList,f)
        pickle.dump(p0V,f)
        pickle.dump(p1V,f)
        pickle.dump(pAb,f)
    return myVocabList, p0V,p1V,pAb


def test(testpath):
    with open('./data/model.pkl','rb') as f:
        myVocabList = pickle.load(f)
        p0V = pickle.load(f)
        p1V = pickle.load(f)
        pAb = pickle.load(f)
    A = 0
    B = 0
    C = 0
    D = 0
    acc = 0
    all = 0
    with open(testpath,'r') as fileobejct:
        for line in fileobejct.readlines():
            label = int(line[:1])
            line = line[1:]
            line = cleanSentence(line)
            line = line.split()
            thisDoc = np.array(bagOfWords2VecMN(myVocabList, line))
            prelabel = classifyNB(thisDoc,p0V,p1V,pAb)
            if label==1:
                all += 1
                if label==prelabel:
                    A += 1
                    acc += 1
                else:
                    C += 1

            if label==0:
                all += 1
                if label==prelabel:
                    D += 1
                    acc += 1
                else:
                    B += 1


    precision = float(A)/(A+B)
    recall = float(A)/(A+C)
    acc = float(acc)/all
    F1 = recall*precision*2/(recall+precision)
    print('precision=',precision)
    print('recall=',recall)
    print('accuracy=',acc)
    print('F1=',F1)


if __name__ == '__main__':
    test('./data/test_file.txt')
    #myVocabList, p0V, p1V, pAb = trainModel()

