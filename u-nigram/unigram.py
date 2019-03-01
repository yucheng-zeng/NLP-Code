import re
import math

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

# 定义UnigramLanguageModel语言模型类
class UnigramLanguageModel:
    # 初始化
    def __init__(self, sentences, smoothing=False):
        self.unigram_frequencies = dict()  # 定义字典,记录单词在语料库中的出现次数
        self.corpus_length = 0  # 记录语料库的长度
        for sentence in sentences:  # 遍历语料库的句子
            for word in sentence:  # 遍历句子中的每一个单词
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:  # 如果这个单词不等于开始标志或者结束标志,语料库长度+1
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2  # 获取一共有多少个不同样的单词,减去2是因为有开始标志或者结束标志
        self.smoothing = smoothing  # 记录是否平滑

    # 计算一元单词出现概率
    def calculate_unigram_probability(self, word):
        word_probability_numerator = self.unigram_frequencies.get(word, 0)  # 获取分子,即该单词在语料库中出现的次数
        word_probability_denominator = self.corpus_length  # 获取分母, 即语料库的长度
        # 判断是否进行平滑处理(这里使用拉普拉斯平滑)
        if self.smoothing:
            word_probability_numerator += 1  # 分子+1
            # add one more to total number of seen unique words for UNK - unseen events
            word_probability_denominator += self.unique_words   # 分母加附加观测值V, 即不同的单词的个数
        return float(word_probability_numerator) / float(word_probability_denominator)  # 计算该单词的概率

    # 在一元语言模型下, 句子出现的概率
    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        '''
        :param sentence: 目标句 子
        :param normalize_probability: 是否规范化概率
        :return: 句子出现的概率
        '''
        sentence_probability_log_sum = 0  # 记录这个句子概率
        for word in sentence:  # 遍历这句子中每一个单词
            if word != SENTENCE_START and word != SENTENCE_END:  # 若果该单词不等于开始标志或者结束标志
                word_probability = self.calculate_unigram_probability(word)  # 计算这个单词在语料库中出现的概率
                # 句子概率加上单词概率, 用math.log(word_probability, 2)防止下溢出
                sentence_probability_log_sum += math.log(word_probability, 2)
        if normalize_probability:
            return math.pow(2, sentence_probability_log_sum)
        else:
            return sentence_probability_log_sum


    # 词典排序
    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())  # 获取全部单词
        full_vocab.remove(SENTENCE_START)  # 去除开始标志, 以便排序
        full_vocab.remove(SENTENCE_END)  # 去除结束标志, 以便排序
        full_vocab.sort()  # 排序
        full_vocab.append(UNK)  # 增加UNK, 用于标志未出现的单词
        full_vocab.append(SENTENCE_START)  # 增加开始标志
        full_vocab.append(SENTENCE_END)  # 增加结束标志
        return full_vocab  # 返回排序之后的全部单词


class BigramLanguageModel(UnigramLanguageModel):
    # 初始化
    def __init__(self, sentences, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, smoothing)  # 初始化unigram模型
        self.bigram_frequencies = dict()  # 定义字典,记录前后单词对在语料库中的出现次数
        self.unique_bigrams = set()  # 记录不相同的单词对
        for sentence in sentences:  # 遍历语料库的句子
            previous_word = None  # 初始化, 句子第一个单词的前一个单词为None
            for word in sentence:  # 遍历句子中的每一个单词
                if previous_word != None:  # 如果前一个单词不为None, 单词对出现次数+1
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))  # 若果前一个单词不等于开始标志或者结束标志,单词对出现次数+1
                previous_word = word  # 将当前单词设置为前一个单词
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)  # 记录一共有多少个不相同单词对

    # 计算单词对出现概率
    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)  # 计算该单词对在语料库中出现的次数
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)  # 计算前一个单词在语料库中出现的次数
        # 判断是否进行平滑处理(这里使用拉普拉斯平滑)
        if self.smoothing:
            bigram_word_probability_numerator += 1  # 分子+1
            bigram_word_probability_denominator += self.unique__bigram_words  # 分母加观测值, 既不相同的单词对的个数
        if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0:
            return 0.0
        else:
            return float(bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    # 计算在二元语言模型, 句子出现的概率
    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        '''
        :param sentence: 目标句子
        :param normalize_probability: 是否规范化概率
        :return: 句子出现的概率
        '''
        bigram_sentence_probability_log_sum = 0  # 记录这个句子概率
        previous_word = None  # 初始化前一个单词
        for word in sentence:  # 遍历这句子中每一个单词
            if previous_word != None:  # 若果前面一个单词不等于None
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)  # 计算这个单词对出现概率
                # 句子概率加上单词概率, 用math.log(word_probability, 2)防止下溢出
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word  # 将当前单词设置为前一个单词
        if normalize_probability:  # 返回规范化概率
            return math.pow(2,bigram_sentence_probability_log_sum)
        else:
            return bigram_sentence_probability_log_sum


# 计算一元语法下一共有多少个单词
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        # remove two for <s> and </s>
        unigram_count += len(sentence) - 2
    return unigram_count


# 计算二元语法下一共有多少个单词
def calculate_number_of_bigrams(sentences):
    bigram_count = 0
    for sentence in sentences:
        # remove one for number of bigrams in sentence
        bigram_count += len(sentence) - 1
    return bigram_count


# 打印一元语法概率表
def print_unigram_probs(sorted_vocab_keys, model):
    for vocab_key in sorted_vocab_keys:  # 遍历单词表
        if vocab_key != SENTENCE_START and vocab_key != SENTENCE_END:  # 若果单词不等于开始标志或者结束标志, 则打印其概率
            print("{}: {}".format(vocab_key if vocab_key != UNK else "UNK",  # 计算打印单词本身或者是未出现词
                                  model.calculate_unigram_probability(vocab_key)), end=" ")  # 计算该单词概率
    print("")

# 打印二元语法概率表
def print_bigram_probs(sorted_vocab_keys, model):
    print("\t\t", end="")
    for vocab_key in sorted_vocab_keys:  # 遍历单词表
        if vocab_key != SENTENCE_START:  # # 若果单词不等于开始标志, 则打印, 相当于打印行表头
            print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")
    print("")
    for vocab_key in sorted_vocab_keys:  # 遍历单词表, 获取前一个单词
        if vocab_key != SENTENCE_END:  # 若果不等于结束标志
            print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")  # 打印, 相当于行表头
            for vocab_key_second in sorted_vocab_keys:  # 遍历单词表, 获取后一个单词
                if vocab_key_second != SENTENCE_START:  # 后一个单词不等于开始标志
                    # 计算概率
                    print("{0:.5f}".format(model.calculate_bigram_probabilty(vocab_key, vocab_key_second)), end="\t\t")
            print("")
    print("")


# 计算一元语法模型的困惑度
def calculate_unigram_perplexity(model, sentences):
    unigram_count = calculate_number_of_unigrams(sentences)  # 计算句子中的单词数
    sentence_probability_log_sum = 0  # 句子出现概率
    for sentence in sentences:
        try:
            # 计算每一个句子的困惑度, 负号代表取分数, 累加求和
            sentence_probability_log_sum -= math.log(model.calculate_sentence_probability(sentence), 2)
        except:
            # 若果溢出, 加上负无穷
            sentence_probability_log_sum -= float('-inf')
    return math.pow(2, sentence_probability_log_sum / unigram_count)  # 除以所有句子的单词数,防止溢出


# 计算二元语法模型的困惑度
def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences) # 计算句子中的单词数
    bigram_sentence_probability_log_sum = 0  # 句子出现概率
    for sentence in sentences:
        try:
            # 计算每一个句子的困惑度, 负号代表取分数, 累加求和
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence), 2)
        except:
            # 若果溢出, 加上负无穷
            bigram_sentence_probability_log_sum -= float('-inf')
    return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)  # 除以所有句子的单词数,防止溢出


if __name__ == '__main__':
    '''
    toy_dataset = read_sentences_from_file("./sampledata.txt")  # 读取训练数据集
    toy_dataset_test = read_sentences_from_file("./sampletest.txt")  # 读取测试数据集

    toy_dataset_model_unsmoothed = BigramLanguageModel(toy_dataset)  # 非平滑二元模型
    toy_dataset_model_smoothed = BigramLanguageModel(toy_dataset, smoothing=True)  # 平滑二元模型

    sorted_vocab_keys = toy_dataset_model_unsmoothed.sorted_vocabulary()  # 获取排序之后的单词表

    print("---------------- Toy dataset ---------------\n")
    print("=== UNIGRAM MODEL ===")
    print("- Unsmoothed  -")
    print_unigram_probs(sorted_vocab_keys, toy_dataset_model_unsmoothed)
    print("\n- Smoothed  -")
    print_unigram_probs(sorted_vocab_keys, toy_dataset_model_smoothed)

    print("")

    print("=== BIGRAM MODEL ===")
    print("- Unsmoothed  -")
    print_bigram_probs(sorted_vocab_keys, toy_dataset_model_unsmoothed)
    print("- Smoothed  -")
    print_bigram_probs(sorted_vocab_keys, toy_dataset_model_smoothed)

    print("")

    print("== SENTENCE PROBABILITIES == ")
    longest_sentence_len = max([len(" ".join(sentence)) for sentence in toy_dataset_test]) + 5
    print("sent", " " * (longest_sentence_len - len("sent") - 2), "uprob\t\tbiprob")
    for sentence in toy_dataset_test:  # 遍历测试集中的句子, 计算其概率
        sentence_string = " ".join(sentence)
        print(sentence_string, end=" " * (longest_sentence_len - len(sentence_string)))
        print("{0:.5f}".format(toy_dataset_model_smoothed.calculate_sentence_probability(sentence)), end="\t\t")  # 计算一元语法的句子概率
        print("{0:.5f}".format(toy_dataset_model_smoothed.calculate_bigram_sentence_probability(sentence)))  # 计算二元语法的句子概率
    print("")

    # 计算困惑度
    print("== TEST PERPLEXITY == ")
    print("unigram: ", calculate_unigram_perplexity(toy_dataset_model_smoothed, toy_dataset_test))
    print("bigram: ", calculate_bigram_perplexity(toy_dataset_model_smoothed, toy_dataset_test))

    print("")

    actual_dataset = read_sentences_from_file("./train.txt")
    actual_dataset_test = read_sentences_from_file("./test.txt")
    actual_dataset_model_smoothed = BigramLanguageModel(actual_dataset, smoothing=True)
    print("---------------- Actual dataset ----------------\n")
    print("PERPLEXITY of train.txt")
    print("unigram: ", calculate_unigram_perplexity(actual_dataset_model_smoothed, actual_dataset))
    print("bigram: ", calculate_bigram_perplexity(actual_dataset_model_smoothed, actual_dataset))

    print("")

    print("PERPLEXITY of test.txt")
    print("unigram: ", calculate_unigram_perplexity(actual_dataset_model_smoothed, actual_dataset_test))
    print("bigram: ", calculate_bigram_perplexity(actual_dataset_model_smoothed, actual_dataset_test))
    '''

    u = [1.38,1.46,1.54,1.66,1.79,1.91,2.09,2.26]
    i = [0.938,0.985,1.050,1.125,1.211,1.293,1.407,1.538]
    u_mean = sum(u)/len(u)
    i_mean = sum(i)/len(i)
    print(u_mean)
    print(i_mean)
    u_square = 0
    i_square = 0
    for u_item in u:
        u_square += u_item*u_item
    for i_item in i:
        i_square += i_item*i_item
    print(u_square/len(u))
    print(i_square/len(i))
    ui = 0
    for index in range(0,len(u)):
        ui += u[index]*i[index]
        print(ui)
    print(ui/len(u))
    result = (1.76*1.19 - 2.1589)/(1.76*1.76-3.19)
    print(result)
    print(1000/0.668)

