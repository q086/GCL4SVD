import sys

dataset = 'FFmpeg'      # FFmpeg ,qemu,FFmpeg+qemu

try:
    least_freq = sys.argv[2]
except:
    least_freq = 1
    print('using default least word frequency = 1')

doc_content_list = []
with open('data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))


word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = doc_content
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = doc_content
    words = temp.split()
    doc_words = []
    for word in words:
        if dataset == 'mr':
            doc_words.append(word)
        elif word_freq[word] >= least_freq:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)


clean_corpus_str = '\n'.join(clean_docs)
# with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
#     f.write(clean_corpus_str)
f  = open('data/corpus/' + dataset + '.clean.txt', 'w',encoding='utf-8')
f.write(clean_corpus_str)

len_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print(len_list)
print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))
