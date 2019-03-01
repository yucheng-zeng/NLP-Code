import CRFPP

# 获取组合词和非组合词中的地名
def tag_line(words):
    chars = []  # 用于记录地名
    tags = []  # 用于几率标签
    temp_word = ''  # 用于合并组合词
    for word in words:
        word = word.strip('\t ')  # 去除前后空行
        w, h = word.split('/')  # 分割组合为 词 标签
        if len(w) == 0:
            continue
        if temp_word == '':
            bracket_start = word.find('[')  # 找到 [ 的下表, 若果没有找打则返回-1
            if bracket_start == -1:  # 未找到括号[，说明不是组合词
                chars.extend(w)
                if h == 'ns':  # 如果这个非组合词是地名的话, 则记录下来, 标记为BME形式或者S
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w)-2) + ['E']
                else:
                    tags += ['O'] * len(w)  # 否则标记为外部词汇
            else:  # 找到了左括号[，进入组合词
                w = w[bracket_start+1:]  # 获取该组合词
                temp_word += w  # 记录组合词内容
        else:
            bracket_end = word.find(']')
            if bracket_end == -1:  # 未找到右括号，仍在组合词中
                temp_word += w  # 记录组合词内容
            else:
                w = temp_word + w
                h = word[bracket_end+1:]  # 组合词结束之后会有标注, 获取组合词的标注
                chars.extend(w)
                if h == 'ns':  # 查看组合词时候是地名, 则记录下来, 标记为BME形式或者S
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w)-2) + ['E']
                else:
                    tags += ['O'] * len(w)  # 否则标记为外部词汇
                temp_word = ''  # 重置组合词

    assert temp_word == ''  # 检测异常错误, 组合词没到 ] 就停止了
    return chars, tags


def corpus_handler(corpus_path):
    with open(corpus_path, encoding='utf8') as corpus_f,\
            open('./data/train.txt', 'w', encoding='utf8') as train_f,\
            open('./data/test.txt', 'w', encoding='utf8') as test_f:
        pos = 0  # 用于划分训练集和测试集
        for line in corpus_f:
            line = line.strip('\r\n\t ')
            if line == '':
                continue
            is_test = True if (pos % 5 == 0) else False  # 20%作为测试集
            words = line.split()[1:]  # 第一列为日期编号，去掉
            if len(words) == 0:
                continue
            line_chars, line_tags = tag_line(words)  # 转换为词 标签结构
            save_obj = test_f if is_test else train_f  # 选择存储目标
            for k, v in enumerate(line_chars):
                save_obj.write(v + '\t' + line_tags[k] + '\n')
            save_obj.write('\n')
            pos += 1


if __name__ == '__main__':
    corpus_handler('data/people-daily.txt')
