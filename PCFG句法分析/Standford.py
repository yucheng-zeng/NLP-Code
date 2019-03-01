import jieba

# PCFG句法分析
from nltk.parse import stanford
import os


def loadmodel(stanford_path,java_path):
    root = stanford_path
    parser_path =  stanford_path + '/stanford-parser.jar'
    model_path =  root + '/stanford-chinese-corenlp-models.jar'


    if not os.environ.get('JAVA_HOME'):
        JAVA_HOME = java_path #'/usr/java/jdk1.8.0_181'
        os.environ['JAVA_HOME'] = JAVA_HOME


    # PCFG模型路径
    pcfg_path = stanford_path+'/stanford-chinese-corenlp/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

    parser = stanford.StanfordParser(
        path_to_jar=parser_path,
        path_to_models_jar=model_path,
        model_path=pcfg_path
    )
    return parser

def start_parse(parser,seg_str):
    sentence = parser.raw_parse(seg_str)
    for line in sentence:
        print(line.leaves())
        print(line)
        line.draw()



if __name__=='__main__':
    string = '我爱北京天安门'
    seg_list = jieba.cut(string, cut_all=False, HMM=True)  # 采用了hmm进行分词
    seg_str = ' '.join(seg_list)
    print(seg_str)

    parser = loadmodel(
        stanford_path='/home/zeng/PycharmProjects/NLP/stanford-tool',
        java_path= '/usr/java/jdk1.8.0_181'
    )
    start_parse(parser,seg_str)





