import pandas as pd
import re
import numpy as np
import time

pd.set_option('display.width', None)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行




def read_raw_dataset_from_csv(dataset_path):
    # pd.read_csv()
    # 参数： sep：指定分隔符，如果不指定参数，则会尝试使用逗号分割
    # error_bad_lines 如果一行包含太多的列。那么默认不会返回DataFrame，
    # 如果设置成false，那么会将改行剔除
    df = pd.read_csv(dataset_path,encoding='ISO-8859-1', header=None,error_bad_lines=False)
    #df = df.fillna('')  #将所有的缺省值替换成指定字符
    return df

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),\+!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



    # cop = re.compile("[^a-zA-Z\s\.]")
    # text = cop.sub("", text)
    # cop = re.compile("\.")
    # text = cop.sub(" ", text)
    # cop = re.compile("\s+")
    # text = cop.sub(" ", text)
    # return text.strip().lower()

    # text = re.sub(r"\'s", " \'s", text)
    # text = re.sub(r"\'ve", " \'ve", text)
    # text = re.sub(r"n\'t", " n\'t", text)
    # text = re.sub(r"\'re", " \'re", text)
    # text = re.sub(r"\'d", " \'d", text)
    # text = re.sub(r"\'ll", " \'ll", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r" ", "", text)
    # text = re.sub(r"!" , " ", text)
    # text = re.sub(r"\(", "  ", text)
    # text = re.sub(r"\)", " ", text)
    # text = re.sub(r"\?", " ", text)
    # text = re.sub(r"//", " ", text)
    # text = re.sub(r"/*", " ", text)
    # text = re.sub(r"\*", " ", text)
    # text = re.sub(r"/\*", " ", text)
    # text = re.sub(r"(1)", " ", text)
    # text = re.sub(r"(2)", " ", text)
    # text = re.sub(r"@"," ", text)
    # text = re.sub(r"}}}", " ", text)
    # text = re.sub(r"{{{", " ", text)
    # text = re.sub(r"\n", " ", text)
    # text= re.sub(r"[^A-Za-z,!?\'\`]", " ", text)


def hd_load_data_and_labels_train(datas):
    i = 0
    x_text = []
    for sent in datas[2]:
        if i == 0:
            i += 1
        else:
            x_text.append(str(sent).strip('\n'))
    self_technical_labels = []
    a = 0
    for data in datas[1]:
        if a == 0:
            a += 1
        else:
            if data != 'WITHOUT_CLASSIFICATION':
                self_technical_labels.append([0,1])
            else:
                self_technical_labels.append([1,0])

    self_technical_labels = np.array(self_technical_labels)
    return [x_text,self_technical_labels]


def hd_load_data_and_labels(datas):
    x_text = [str(sent).strip('\n') for sent in datas[2]]
    self_technical_labels = []
    for data in datas[1]:
        # print(data)
        # print(data.type)
        if data != 0:
            self_technical_labels.append([0,1])
        else:
            self_technical_labels.append([1,0])

    self_technical_labels = np.array(self_technical_labels)
    return [x_text,self_technical_labels]

def konge(datas):
    x_text = []
    index = []
    num = 0
    for i in datas:
        if len(i) == 0:
            index.append(num)
        else:
            x_text.append(i)
        num += 1
    return x_text,index


def load_data_and_labels(datas):
    # 如果数据缺失，np会对缺失的值加入NaN 但是 NaN是 Float类型 这样会对我们处理数据集产生麻烦
    # 从文件中读取数据的时候，如果数据为空，在np中会自动添加NaN
    # 如果你的数据集中的数据都是str类型的，并且你会对数据集进行str类型专有的操作，
    # 那么会出现问题，因为你的数据集中不光有str类型，还有NaN的float类型，因此你需要处理NaN
    # 暴力解法：将数据集中含有NaN的行全部删除
    # data_without_NaN =datas.dropna(axis=0)
    # 普通解法： 可以将NaN转成别的 datas.fillna() 但是转换的数还是float类型

    data_without_NaN = datas.dropna(axis=0)     # 丢弃含有缺失值的行
    x_text = [sent.strip('\n') for sent in data_without_NaN[2]]     # 得到文本内容
    x_text = [clean_str(sent) for sent in x_text]       # 进行clean_str()
    x_text,index = konge(x_text)    # 如果数据集中缺少，文本内容，index就是数据集中对应位置i的列表
    self_technical_labels=[]        # 整个数据集的标签列表
    # file = r'./text'
    # with open(file, 'a+') as f:
    #     for i in data_without_NaN[1]:
    #         f.write(i + '\n')
    # ******* 十分关键 在处理一个数据的过程中出现问题，
    # python中的dataframe提出部分数据后，索引消失就出错误。
    # 是因为我对原始数据删除了部分异常数据导致的。
    # 下面一行代码重新定义索引，才能支持遍历
    data_without_NaN = data_without_NaN.reset_index(drop=True)     # drop=True表示重新设置索引后，删除原索引
    for index_ in range(len(data_without_NaN[1])):
        if index_ in index:     # 如果缺少文本内容，pass
            pass
        else:
            print("data_without_NaN[1][index_]:",data_without_NaN[1][index_])
            if data_without_NaN[1][index_] != 'WITHOUT_CLASSIFICATION':
                self_technical_labels.append([0,1])     # 添加对应的标签，但是，我不懂明明有三种类型，怎么只是区分这个？？还有为什么是[]这种形式的
            else:
                self_technical_labels.append([1,0])
    self_technical_labels = np.array(self_technical_labels)
    return [x_text,self_technical_labels]

def batch_iter(data, batch_size,shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # 需要将list的data转换为array
    # list是列表，list中的元素的数据类型可以不一样。array是数组，数组中的元素的数据类型必须一样。
    data = np.array(data)
    data_size = len(data) # 训练集个数56048
    num_batches_per_epoch = int(data_size/ batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size  # 0
        end_index = min((batch_num + 1) * batch_size, data_size)  # min(64,数据长度大小)
        yield shuffled_data[start_index:end_index]


    # 带有yield的函数不再是一个普通函数，而是一个生成器generator,可用于迭代，工作原理同上。
    # 简要理解：yield就是return返回一个值，并且记录这个返回的位置，下次迭代就从这个位置后（下一行）开始
    # 带有yield的函数不仅仅只用于for循环中，而且可用于某个函数的参数，只要这个函数的参数允许迭代参数。

# 测试集数据处理
def batch_iter_test(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # 需要将list的data转换为array
    # list是列表，list中的元素的数据类型可以不一样。array是数组，数组中的元素的数据类型必须一样。
    data = np.array(data)
    data_size = len(data) # 训练集个数56048
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size  # 0
            end_index = min((batch_num + 1) * batch_size, data_size)  # min(64,数据长度大小)
            yield shuffled_data[start_index:end_index]

def prepare_tf_input_datas(x_text, vocabulary, max_doc_len):
    temp = []
    prepared_datas = []
    words_to_ids,length_of_sentence =data_padding_and_to_ids(x_text, vocabulary, max_doc_len)
    temp += words_to_ids
    temp.append(length_of_sentence)
    prepared_datas.append(temp)
    return prepared_datas

# 针对一个doc的数据填充, 将单词转换成数字索引
def data_padding_and_to_ids(feature, vocabulary, max_doc_len):
    '''
    :param feature:  bugid
    :param vocabulary:
    :param max_doc_len: 400
    :return:
    '''
    # 返回单词在词汇表中的value(or索引), 如果单词不在词汇表, 返回UNK_ID=3
    # ids = [word_vocabulary.get(word, UNK_ID) for word in sentence]
    ids = []
    PAD_ID = 0
    for word in feature:
        try:
            '''vocabulary是字典 word是bugid也是键值对的键
            将键值对的值存入ids中 
            如果出现词汇表中没有的词,忽略
            '''
            ids.append(vocabulary.index(word))
        except ValueError:
            continue                # TODO: OOV，如果真的出现词汇表中没有的单词，忽略,事实上，应该添加上UNK标记才好。
    if len(ids) > max_doc_len:      # 长于max_doc_len的步长都切掉
        ids = ids[:max_doc_len]
    '''注意这里在ids之前填充0, 这样做来避免太长的0影响本身单词的记忆效果'''
    words_as_ids = ids+[PAD_ID] * (max_doc_len - len(ids))      # 用PAD_ID来填充数据
    return words_as_ids, len(ids)



if __name__ == '__main__':
    # 数据集拽入服务器中 不一定是txt形式 需要确认，要不程序会报错
    datas = read_raw_dataset_from_csv('../data/columba.csv')
    load_data_and_labels(datas)