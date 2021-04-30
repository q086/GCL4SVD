from __future__ import division
from __future__ import print_function

import operator
from functools import reduce
from itertools import chain

import time
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
import pickle as pkl
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
    compute_confident_joint,
    estimate_latent,
estimate_confident_joint_and_cv_pred_proba
)
from utils import *
import csv
import warnings
warnings.filterwarnings('ignore')
from run_models import GNN, MLP
import os
import numpy as np
import json_lines
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)
dataset = "qemu"       # FFmpeg , VDSIC, qemu ,FFmpeg+qemu
# Settings
def clean_l():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('label', 0, 'test_label')  # 0 1
    flags.DEFINE_string('dataset',dataset, 'Dataset string.')
    flags.DEFINE_string('model', 'gnn', 'Model string.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
    flags.DEFINE_integer('batch_size', 128, 'Size of batches per epoch.')
    flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
    flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.')  # 32, 64, 96, 128
    flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('evaluateEvery', 100, 'How many steps are run for validation each time.')

    # Load data
    train_adj, train_feature, train_y, d_lossweight = load_data(
        FLAGS.dataset)

    # Some preprocessing
    print('loading training set')
    train_adj, train_mask = preprocess_adj(train_adj)
    train_feature = preprocess_features(train_feature)

    if FLAGS.model == 'gnn':
        # support = [preprocess_adj(adj)]
        # num_supports = 1
        model_func = GNN
    elif FLAGS.model == 'gcn_cheby':  # not used
        # support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GNN
    elif FLAGS.model == 'dense':  # not used
        # support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': tf.placeholder(tf.float32, shape=(None, None, None)),
        'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
        'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
        'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'steps_per_epoch': tf.placeholder_with_default(0.0, shape=()),
        'globalStep': tf.placeholder_with_default(0.0, shape=()),
        'd_lossweight': d_lossweight  # 加权系数
    }

    # label smoothing
    # label_smoothing = 0.1
    # num_classes = y_train.shape[1]
    # y_train = (1.0 - label_smoothing) * y_train + label_smoothing / num_classes

    # Create model
    model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

    # Initialize session

    gpu_options = tf.GPUOptions()
    session_conf = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    # sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    print('train start...')

    # Train model
    currentStep = 0
    steps_per_epoch = (int)(len(train_y) / FLAGS.batch_size) + 1
    train_pred = []
    train_label = []
    for epoch in range(FLAGS.epochs):
        print("epoch:",epoch)
        t = time.time()
        # Training step
        indices = np.arange(0, len(train_y))

        # np.random.shuffle(indices)

        train_loss, train_acc = 0, 0
        for start in range(0, len(train_y), FLAGS.batch_size):
            currentStep += 1
            print("currentStep:", currentStep)
            end = start + FLAGS.batch_size
            idx = indices[start:end]
            # Construct feed dictionary
            feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx], train_y[idx],
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            feed_dict.update({placeholders['steps_per_epoch']: steps_per_epoch})
            feed_dict.update({placeholders['globalStep']: currentStep})

            outs = sess.run(
                [model.opt_op, model.loss, model.accuracy, model.outputs, model.placeholders['labels'], model.labels,model.l_r],
                feed_dict=feed_dict)
            train_preds = outs[3]
            train_labels_argmax = outs[5]
            lr = outs[6]
            print("Epoch:", '%04d' % (epoch + 1), "Step:{}".format(currentStep), "learning_rate=", "{:.8f}".format(lr))
            if epoch == FLAGS.epochs - 1:
                train_pred.append(train_preds)
                train_label.append(train_labels_argmax)

    # 模型的保存
    print("Save model...")
    model.save(sess)
    print(" Finished save model...")

    numpy_array_of_predicted_probabilities = list(chain.from_iterable(train_pred))
    numpy_array_of_noisy_labels = list(chain.from_iterable(train_label))
    print(len(numpy_array_of_predicted_probabilities))
    print(len(numpy_array_of_noisy_labels))

    numpy_array_of_predicted_probabilities = np.array(numpy_array_of_predicted_probabilities)
    numpy_array_of_noisy_labels = np.array(numpy_array_of_noisy_labels)

    print(np.shape(numpy_array_of_noisy_labels))
    print(np.shape(numpy_array_of_predicted_probabilities))

    ordered = get_noise_indices(s=numpy_array_of_noisy_labels, psx=numpy_array_of_predicted_probabilities,
                                sorted_index_method='normalized_margin')        # 标记标签错误

    print(ordered)
    print("去掉数据的数量：",len(ordered))

    corpus = []
    labels = []
    # idx = []
    with open('/dataprocess/'+dataset + '_train.jsonl', 'rb') as ftrain:
        for item in json_lines.reader(ftrain):
            data_str_target = item.get("target")

            text = item.get("func")
            corpus.append(text.strip('\n'))
            labels.append(data_str_target)
    print("text:", len(corpus))
    print("labels:", len(labels))
    # print("idx:", len(idx))
    print("************************************************")
    # count_vec = TfidfVectorizer(binary=False, decode_error='ignore')
    # x = count_vec.fit_transform(corpus)
    # x = x.toarray()
    labels = np.array(labels)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    new_corpus = np.delete(corpus, ordered)  # 删除指定索引，得到新的数组
    new_labels = np.delete(labels,ordered)
    print("text_clean:", len(new_corpus))
    print("labels_clean:", len(new_labels))

    # 新的数组加入写入到json文件中
    name_list = ['target', 'func']
    total_list = []

    for i in range(len(new_corpus)):
        val_list = []
        # val_list.append(str(new_labels[i]))
        val_list.append(str(new_labels[i]))
        val_list.append(str(new_corpus[i]))
        dict_single = dict(zip(name_list, val_list))
        total_list.append(dict_single)
    with open('/data_afterGCL/'+dataset + '_train_clean.jsonl', 'w') as f:
        for idx, js in enumerate(total_list):
            f.write(json.dumps(js) + '\n')

if __name__=="__main__":
    clean_l()



