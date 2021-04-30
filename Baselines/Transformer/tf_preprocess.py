import csv
import re
# 将RNN的输入格式改为Transformer的输入格式
csv_file_train = open('./data/FFmpeg+qemu_train.csv',encoding='ISO-8859-1')
csv_file_test = open('./data/FFmpeg+qemu_test.csv',encoding='ISO-8859-1')
csv_file_valid = open('./data/FFmpeg+qemu_valid.csv',encoding='ISO-8859-1')
csv_reader_lines_train = csv.reader(csv_file_train)
csv_reader_lines_test = csv.reader(csv_file_test)
csv_reader_lines_val = csv.reader(csv_file_valid)
labels = ['0','1']


train_csv_file = open('./data/preProcess/tf_FFmpeg+qemu_train.csv','w',newline='')
test_csv_file = open('./data/preProcess/tf_FFmpeg+qemu_test.csv','w',newline='')
val_csv_file = open('./data/preProcess/tf_FFmpeg+qemu_val.csv','w',newline='')
writer_train = csv.writer(train_csv_file)
writer_test = csv.writer(test_csv_file)
writer_val = csv.writer(val_csv_file)
writer_train.writerow(['review', 'sentiment'])
writer_test.writerow(['review', 'sentiment'])
writer_val.writerow(['review', 'sentiment'])

train_corpus = []
train_corpus_label = []
train_data = []
for one_line in csv_reader_lines_train:
    one_line_text = one_line[2]
    text = ''
    for alp in one_line_text:
        text +=alp.strip('\n')
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


test_corpus = []
test_corpus_label = []
test_data = []
for one_line in csv_reader_lines_test:
    one_line_text = one_line[2]
    text = ''
    for alp in one_line_text:
        text += alp.strip('\n')
    test_corpus.append(one_line_text.strip('\n'))
    if one_line[1] == '0':
        test_corpus_label.append(labels[0])
        data_line = [text,labels[0]]
        test_data.append(data_line)
    else:
        test_corpus_label.append(labels[1])
        data_line = [text, labels[1]]
        test_data.append(data_line)
writer_test.writerows(test_data)
test_csv_file.close()

val_corpus = []
val_corpus_label = []
val_data = []
for one_line in csv_reader_lines_val:
    one_line_text = one_line[2]
    text = ''
    for alp in one_line_text:
        text += alp.strip('\n')
    val_corpus.append(one_line_text.strip('\n'))
    if one_line[1] == '0':
        val_corpus_label.append(labels[0])
        data_line = [text,labels[0]]
        val_data.append(data_line)
    else:
        val_corpus_label.append(labels[1])
        data_line = [text, labels[1]]
        val_data.append(data_line)
writer_val.writerows(val_data)
val_csv_file.close()
