#-*-coding:utf-8-*-
import csv
import json
import json_lines
import sys
import codecs
import re

# 就是将总的数据集首先变成RNN的数据集的形式，
dataset = 'FFmpeg+qemu'
filename = 'FFmpeg+qemu'     # FFmpeg_train.jsonl,FFmpeg_test.jsonl,FFmpeg_valid.jsonl
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string.replace("\n", "")
    string.replace("->"," -> ")

    string = re.sub(r"[^A-Za-z0-9()%/\\:+\->&\[\]|=<*>.,_{};!?\'\`]", " ", string)      # 实现正则的替换，^匹配开始位置，匹配数字，字母，下划线，（）
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"<<", " << ", string)
    string = re.sub(r">>", " >> ", string)
    string = re.sub(r"&", " & ", string)
    string = re.sub(r"/\*", " /* ", string)
    # string = re.sub(r"->", " -> ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\*/", " */ ", string)
    string = re.sub(r"\*", " * ", string)
    string = re.sub(r"&&", " && ", string)  # 新添加的&&
    string = re.sub(r":", " : ", string)
    # string = re.sub(r"\||", " || ", string)    # 新添加的||
    string = re.sub(r";", " ; ", string)    # 新添加的；

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\{", " { ", string)    # 新添加的{
    string = re.sub(r"}", " } ", string)    # 新添加的}
    string = re.sub(r"'", " ' ", string)    # 新添加的'
    string = re.sub(r"\+", " + ", string)    # 新添加的+
    string = re.sub(r"-", " - ", string)       # 新添加的->
    string = re.sub(r">", " > ", string)
    string = re.sub(r"<", " < ", string)
    string = re.sub(r"\s{2,}", " ", string)     # 匹配任意空白字符，2到无穷次，只用一个‘ ’替换掉

    return string.strip()
# 得到三个列表
dataset_list = []
label_list = []
func_list = []
with open(filename+'_train.jsonl', 'rb') as frain:
    for item in json_lines.reader(frain):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        data_str = ' '.join(func_str.split()[:1200])
        dataset_list.append(dataset)
        label_list.append(item.get("target"))
        func_list.append(data_str)
with open(filename+'_test.jsonl', 'rb') as fest:
    for item in json_lines.reader(fest):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        data_str = ' '.join(func_str.split()[:1200])
        dataset_list.append(dataset)
        label_list.append(item.get("target"))
        func_list.append(data_str)
with open(filename+'_valid.jsonl', 'rb') as fvalid:
    for item in json_lines.reader(fvalid):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        data_str = ' '.join(func_str.split()[:1200])
        dataset_list.append(dataset)
        label_list.append(item.get("target"))
        func_list.append(data_str)

total_list = []
# 将三个列表进行合并，创建一个新列表
for i in range(len(dataset_list)):
    for j in range(len(label_list)):
        if j == i:
            for k in range(len(func_list)):
                if j == k:
                    t = (dataset_list[i],label_list[j],func_list[k])
                    total_list.append(t)
# 新的列表存入csv文件
with open(filename+".csv",'w+',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(total_list)

# -------------------------------------------------
csv_file_train = open('FFmpeg+qemu.csv',encoding='ISO-8859-1')
csv_reader_lines_train = csv.reader(csv_file_train)
labels = ['0','1']


train_csv_file = open('./data/preProcess/tf_FFmepg+qemu.csv','w',newline='')
writer_train = csv.writer(train_csv_file)
writer_train.writerow(['review', 'sentiment'])


train_corpus = []
train_corpus_label = []
train_data = []
for one_line in csv_reader_lines_train:
    one_line_text = one_line[2]
    text = ''
    for alp in one_line_text:
        text += alp.strip('\n')
    train_corpus.append(one_line_text.strip('\n'))
    if one_line[1] == '0':
        train_corpus_label.append(labels[0])
        data_line = [text,labels[0]]
        train_data.append(data_line)
    else:
        train_corpus_label.append(labels[1])
        data_line = [text, labels[1]]
        train_data.append(data_line)
writer_train.writerows(train_data)
train_csv_file.close()
