import os

# 读取新闻文件夹
def listdir(path):
    import os
    filelist = []
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        filelist.append(os.path.join(path, file))
    return filelist


# 将文件夹里面的文件内容读取出来,按比例切分测试集以及语料库集
def CreateCorpus(path):
    filelist = []
    for file in os.listdir(path):
        filelist.append(os.path.join(path, file))
    i = 0
    with open(path + '/corpus.txt', 'w') as corpus,\
            open(path + '/test.txt', 'w') as test:
        for filename in filelist:
            with open(filename,'r',encoding='utf-8-sig') as fileobject:
                content = ''
                for line in fileobject.readlines():
                    line = line.strip()
                    line = line.replace('&nbsp','')
                    content += line
                save_object = corpus
                if i%10==0:
                    save_object = test
                save_object.write(content+'\n')
                i += 1

if __name__=='__main__':
    filelist = listdir('./news')
    print(filelist)
    for folder in filelist:
        if os.path.exists(folder):
            CreateCorpus(folder)