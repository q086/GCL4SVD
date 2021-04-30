import json_lines
import json
import re
import sys

def count_words(aaa):
    str_list = aaa.split()
    return len(str_list)

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


line_num = -1
dataset = "FFmpeg"  # FFmpeg, qemu, FFmpeg+qemu

fw = open('../data/corpus/'+ dataset+'.txt', 'w+', encoding='utf-8')
fl = open('../data/'+dataset+'.txt', 'w+', encoding='utf-8')

with open(dataset+'_train.jsonl', 'rb') as frain:
    for item in json_lines.reader(frain):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        if item.get("target") == 0:
            line_num = line_num + 1
            label_str = str(line_num) + "\ttrain\t0\n"
            fl.write(label_str)
            data_str = ' '.join(func_str.split()[:1200]) + "\n"
            fw.write(data_str)
        else:
            line_num = line_num + 1
            label_str = str(line_num) + "\ttrain\t1\n"
            fl.write(label_str)
            data_str = ' '.join(func_str.split()[:1200]) + "\n"
            fw.write(data_str)
print(line_num+1)
fw.close()
fl.close()
