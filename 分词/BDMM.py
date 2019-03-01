
class BDMM(object):
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

    def RMM_cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximun, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index - size):index]
                if piece in self.disctionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result[::-1]

    def MM_cut(self, text):
        result = []
        index = 0
        length = len(text)
        while index < length:
            word = None
            for size in range(self.maximun, 0, -1):
                if index + size > length:
                    size = length - index
                piece = text[index:index + size]
                if piece in self.disctionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                index += 1
        return result

    def cut(self,text):
        result1 = self.MM_cut(text)
        result2 = self.RMM_cut(text)
        if len(result1)>len(result2):
            return result2
        else:
            return result1
if __name__=='__main__':
    tokenizer = BDMM()
    text = '南京市长江大桥'
    result = tokenizer.cut(text)
    print(result)