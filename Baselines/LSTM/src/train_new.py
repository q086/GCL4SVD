import sys
sys.path.append(r"D:\课题\GCL4SVD\Baselines\rnn")  # 绝对路径
import tensorflow as tf
import numpy as np
import os
from utils import read_raw_dataset_from_csv
from RNN_new import RNN
from utils import batch_iter_test,batch_iter
from utils import hd_load_data_and_labels
from sklearn.metrics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.flags.DEFINE_string(
    "datas", "../data/FFmpeg+qemu_train.csv", "Path of data")
tf.flags.DEFINE_string(
    "datas_test", "../data/FFmpeg+qemu_test.csv", "Path of positive data")
tf.flags.DEFINE_string(
    "datas_val", "../data/FFmpeg+qemu_valid.csv", "Path of positive data")
tf.flags.DEFINE_string('save_file', 'FFmpeg+qemu-RNN.txt', 'file String.')

tf.flags.DEFINE_integer("max_sentence_length", 100,
                        "Max sentence length in test/test data (Default: 100)")     # 最大序列长度！！！
# Model Hyperparameters
tf.flags.DEFINE_string(
    "cell_type", "lstm", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_string(
    "word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("embedding_dim", 300,
                        "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer(
    "hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda",0.2,
                      "L2 regularization lambda (Default: 3.0)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_float("dev_sample_percentage", .1,
                        "train.split")      # 用于划分验证集的
# Training parameters
tf.flags.DEFINE_integer("batch_size",128, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer(
    "num_epochs", 300, "Number of training epochs (Default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "Which learning rate to start with. (Default: 1e-3)")


FLAGS = tf.flags.FLAGS

f = open(FLAGS.save_file,'w+',encoding='utf-8')

def train():
    # 这里再加上直接输入的验证集
    datas = read_raw_dataset_from_csv(FLAGS.datas)      # 训练集的数据
    datas_test = read_raw_dataset_from_csv(FLAGS.datas_test)    # 测试集的数据
    datas_val = read_raw_dataset_from_csv(FLAGS.datas_val)
    # x_text,y_train =hd_load_data_and_labels_train(datas)
    print("开始加载数据！！！！！！！！！！")
    x_text, y_train = hd_load_data_and_labels(datas)
    x_test,y_test = hd_load_data_and_labels(datas_test)
    x_val, y_val = hd_load_data_and_labels(datas_val)

    # 创建词汇表
    max_sentence_length = min(FLAGS.max_sentence_length, max([len(x.split()) for x in x_text]))
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=max_sentence_length)
    x_train = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(
        len(text_vocab_processor.vocabulary_)))
    print("x = {0}".format(x_train.shape))  # 文本矩阵大小
    print("y = {0}".format(y_train.shape))  # 标签矩阵大小

    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    max_f1 = 0
    min_loss = 10000000000000000


    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1], # 词长度 根据设置的最大词长度 100
                num_classes=y_train.shape[1], # 2
                vocab_size=len(text_vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                rnn.loss, global_step=global_step)
        

            # test部分
            x_test1 = np.array(list(text_vocab_processor.fit_transform(x_test)))
            y_test_idx = np.argmax(y_test, axis=1)  # axis =1 表示行

            # val部分
            x_val1 = np.array(list(text_vocab_processor.fit_transform(x_val)))
            y_val_idx = np.argmax(y_val, axis=1)  # axis =1 表示行


            # 初始化变量
            sess.run(tf.global_variables_initializer())

            f1 = 0
            batches = batch_iter_test(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, train_loss, train_accuracy, logits, y = sess.run(
                    [train_op, global_step, rnn.loss,
                     rnn.accuracy, rnn.logits, rnn.input_y], feed_dict)


                if step % FLAGS.evaluate_every == 0:
                    feed_dict_val = {
                        rnn.input_text: x_val1,
                        rnn.input_y: y_val,
                        rnn.dropout_keep_prob: 1.0
                    }
                    val_loss, val_accuracy, val_logits, val_predictions = sess.run(
                        [rnn.loss, rnn.accuracy, rnn.logits, rnn.predictions], feed_dict_val)
                    val_pre = precision_score(y_val_idx, val_predictions, average='binary', pos_label=1)
                    val_recall = recall_score(y_val_idx, val_predictions, average='binary', pos_label=1)
                    val_f1_measure = f1_score(y_val_idx, val_predictions, average='binary', pos_label=1)

                    # if val_loss <= min_loss:
                    if val_f1_measure >= max_f1:
                        # min_loss = val_loss
                        max_f1 = val_f1_measure

                        feed_dict_dev = {
                            rnn.input_text: x_test1,
                            rnn.input_y: y_test,
                            rnn.dropout_keep_prob: 1.0
                        }
                        test_loss, test_accuracy, logits, test_predictions = sess.run(
                            [rnn.loss, rnn.accuracy, rnn.logits, rnn.predictions], feed_dict_dev)
                        test_pre = precision_score(y_test_idx, test_predictions, average='binary', pos_label=1)
                        test_recall = recall_score(y_test_idx, test_predictions, average='binary', pos_label=1)
                        f1_measure = f1_score(y_test_idx, test_predictions, average='binary', pos_label=1)
                        # auc = roc_auc_score(y_test_idx, test_predictions)
                        test_mcc = matthews_corrcoef(y_test_idx,test_predictions)


                        if f1_measure >= float(f1):
                            f1 = f1_measure

                        print("train_loss=", "{:.5f}".format(train_loss),
                              "val_loss=", "{:.5f}".format(val_loss), "val_f1=", "{:.3f}".format(val_f1_measure), "test_loss=", "{:.5f}".format(test_loss),
                              "test_accuracy=", "{:.3f}".format(test_accuracy),
                              "test_pre=", "{:.3f}".format(test_pre), "test_recall=",
                              "{:.3f}".format(test_recall), "test_f1=", "{:.3f}".format(f1_measure),
                               "best_f1=", "{:.3f}".format(f1),"test_mcc=", "{:.3f}".format(test_mcc))

                        result = " train_loss=" + "{:.5f} ".format(
                            train_loss) + " val_loss=" + "{:.5f}".format(val_loss) + " val_f1=" + "{:.3f}".format(
                            val_f1_measure) + "test_loss=" + "{:.5f}".format(test_loss)+ "test_accuracy=" + "{:.3f}".format(test_accuracy) +  " test_pre=" + "{:.3f} ".format(
                            test_pre) + " test_recall=" + "{:.3f} ".format(
                            test_recall) + " test_f1=" + "{:.3f} ".format(
                            f1_measure) + " best_f1=" + "{:.3f} ".format(
                            f1)+"test_mcc=" + "{:.3f}".format(test_mcc)
                        f.write(result)
                        f.write('\n')
    f.write('end')





def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()