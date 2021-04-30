from gensim.models import word2vec
import logging
from util.data_preprocess import *
import os

def train_word2vec(datas):
    x_text, y_label = load_data_and_labels(datas)
    sentences = [s.split() for s in x_text]
    sentences1 = []
    for i in sentences:
        sentences1.append(i)
    t1 = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sg=1 skip-gram模型 iter = 8 迭代数据集8次
    model = word2vec.Word2Vec(sentences1,size=300,sg=1, iter=8,min_count=1)
    model.wv.save_word2vec_format('tech_word2vec.txt',binary=False)
    # model.wv.save_word2vec_format('tech_word2vec.bin', binary=True)
    # path = 'word2vec.model'
    # if os.path.exists(path):
    #     os.remove(path)
    # model.save(path)
    # print("--------------------------------------------")
    print("Training word2vec model cost %.3f seconds...\n" % (time.time()-t1))

def read_embedding_model(model_path):
    word_vecs = word2vec.Word2Vec.load(model_path)
    # print(word_vecs.wv.vectors) 这个实际上是ndarray格式的lookup_table *** 词典中的每个词对应的词汇表
    # print(word_vecs.wv.index2word) 每个index对应的词,这个实际上就相当于vocabulary
    # print(word_vecs.wv.vocab) 查看word和vector的对应关系
    # print(word_vecs.wv.vectors.shape)
    return word_vecs.wv.index2word,word_vecs.wv.vectors,word_vecs

if __name__=='__main__':
    # vector_word_filename = './word2.txt'
    datas = read_raw_dataset_from_csv('../data/technical_debt_dataset.csv')
    train_word2vec(datas)
    # read_embedding_model('word2vec.model')

