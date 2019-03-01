'''
def Reverse_dictionary(filename1,filename2):
    dictionary = set()
    with open(filename1, 'r') as fileobject:
        for line in fileobject:
            line = line.strip()
            if line:
                dictionary.add(line)
    with open(filename2,'w') as fileobject:
        for line in dictionary:
            fileobject.write(line[::-1]+'\n')
'''
class RMM(object):
    def __init__(self):
        self.disctionary = set()
        self.maximun = 0
        with open('simple_dict.txt', 'r') as fileobject:
            for line in fileobject:
                line = line.strip()
                if line:
                    self.disctionary.add(line)
                    if len(line) > self.maximun:
                        self.maximun = len(line)

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximun, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index-size):index]
                if piece in self.disctionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result[::-1]


if __name__=='__main__':
    text = '南京市长江大桥'
    tokenizer = RMM()
    result = tokenizer.cut(text)
    print(result)