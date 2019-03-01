import os


def split_train_test(pos_path,neg_path):
    pos_list = os.listdir(pos_path)
    neg_list = os.listdir(neg_path)
    i = 0
    with open('./data/pos_file.txt','w') as pos_object,\
            open('./data/neg_file.txt','w') as neg_object,\
            open('./data/test_file.txt','w') as test_object:
        for pos_name in pos_list:
            full_pos_name = pos_path+'/'+pos_name
            with open(full_pos_name,'r') as fileobjcet:
                if i % 100 == 0:
                    for line in fileobjcet.readlines():
                        line = line.strip()
                        test_object.write('1 '+ line + '\n')

                else:
                    for line in fileobjcet.readlines():
                        line = line.strip()
                        pos_object.write(line + '\n')

            i += 1
        for neg_name in neg_list:
            full_neg_name = neg_path+'/'+neg_name
            with open(full_neg_name, 'r') as fileobjcet:
                if i % 100 == 0:
                    for line in fileobjcet.readlines():
                        line = line.strip()
                        test_object.write('0 ' + line + '\n')

                else:
                    for line in fileobjcet.readlines():
                        line = line.strip()
                        neg_object.write(line + '\n')

            i += 1



if __name__=='__main__':
    split_train_test('../pos','../neg')
