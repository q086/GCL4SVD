#-*-coding:utf-8-*-
import csv
import json
import json_lines
import sys
import codecs
import re

dataset = 'FFmpeg+qemu'
filename = 'FFmpeg+qemu_valid'     # FFmpeg_train.jsonl,FFmpeg_test.jsonl,FFmpeg_valid.jsonl
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
with open(filename+'.jsonl', 'rb') as frain:
    for item in json_lines.reader(frain):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        # func_str = item.get("func")
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
with open('../data/'+filename+".csv",'w+',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(total_list)