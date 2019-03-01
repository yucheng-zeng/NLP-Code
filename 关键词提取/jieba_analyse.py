import jieba.analyse

jieba.load_userdict('./user_dict.txt')

def load_stopword(filename):
    stop_word = []
    with open(filename,'r') as fileobject:
        for line in fileobject.readlines():
            stop_word.append(line.strip())
    return stop_word

def load_text(filename):
    text = ''
    with open(filename, 'r') as fileobject:
        for line in fileobject.readlines():
            text += line
    return text

def load_sentence(filename):
    text = []
    with open(filename, 'r') as fileobject:
        for line in fileobject.readlines():
            line = line.strip()
            line = line.split('。')
            for i in line:
                if len(i) != 0:
                    text.append(i)
    return text

def split_sentence(text):
    result = []
    for sent in text:
        result.append(list(jieba.cut(sent, cut_all=False)))
    return result

def remove_stopword(text):
    stop_word = load_stopword('stop_word.txt')
    print(stop_word)
    for sentence in text:
        for word in sentence:
            print(word)
            if word in stop_word:
                print(1)
                while word in sentence:
                    sentence.remove(word)
        if len(sentence) == 0:
            text.remove([])#删除空行


def TF_IDF(text):
    word_bag = dict()
    sentence_num = len(text)
    for sentence in text:
        for word in sentence:
            # TF
            word_bag[word] = word_bag.get(word,0) + 1



if __name__=='__main__':

    sentence = load_sentence('10.txt')
    print(sentence)
    result = split_sentence(sentence)
    remove_stopword(result)
    print(result)

    stop_word = load_stopword('stop_word.txt')
    print(stop_word)
    print('、' in stop_word)
    print('\u3000' in stop_word)


