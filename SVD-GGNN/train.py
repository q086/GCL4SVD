from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf
from sklearn import metrics
import pickle as pkl
from sklearn.metrics import f1_score,precision_score,recall_score,matthews_corrcoef
from utils_wq import *
from models import GNN, MLP
import os
# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

dataset = "FFmpeg"        # FFmpeg , qemu ,FFmpeg+qemu
# dataset = sys.argv[1]
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
flags.DEFINE_string('save_file', dataset+'_result.txt', 'file String.')
flags.DEFINE_string('model', 'gnn', 'Model string.') 
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 128, 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('hidden', 512, 'Number of units in hidden layer.')      # 128 ,256, 512
flags.DEFINE_integer('steps', 6, 'Number of graph layers.') # 1, 2, 4, 6
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('evaluateEvery',50 , 'How many steps are run for validation each time.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # Not used

# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(FLAGS.dataset)

f = open(FLAGS.save_file,'w+',encoding='utf-8')
# Some preprocessing
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)


if FLAGS.model == 'gnn':
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    model_func = GNN
elif FLAGS.model == 'gcn_cheby': # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN
elif FLAGS.model == 'dense': # not used
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
    'num_features_nonzero': tf.placeholder(tf.int32),# helper variable for sparse dropout
    'steps_per_epoch': tf.placeholder_with_default(0.0, shape=()),
    'globalStep': tf.placeholder_with_default(0.0, shape=())
}


# label smoothing
# label_smoothing = 0.1
# num_classes = y_train.shape[1]
# y_train = (1.0 - label_smoothing) * y_train + label_smoothing / num_classes


# Create model
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# Initialize session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
session_conf = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=session_conf)

# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('logs/', sess.graph)

# Define model evaluation function
def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels,model.l_r], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4], outs_val[5]


# Init variables
sess.run(tf.global_variables_initializer())


best_epoch =0
max_f1 = 0
End_test_f1 = 0
End_test_acc = 0
End_test_precision =0
End_test_recall = 0
End_test_mcc = 0
print('train start...')
# Train model
currentStep = 0
steps_per_epoch = (int)(len(train_y)/FLAGS.batch_size)+1
for epoch in range(FLAGS.epochs):
    t = time.time()
        
    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)
    
    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), FLAGS.batch_size):
        currentStep += 1
        end = start + FLAGS.batch_size
        idx = indices[start:end]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx], train_y[idx], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['steps_per_epoch']: steps_per_epoch})
        feed_dict.update({placeholders['globalStep']: currentStep})

        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.l_r], feed_dict=feed_dict)
        train_loss = outs[1]
        train_acc = outs[2]
        lr = outs[3]


        print("Epoch:", '%04d' % (epoch + 1), "Step:{}".format(currentStep), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc),"learning_rate=", "{:.8f}".format(lr))

        if currentStep % FLAGS.evaluateEvery == 0:
            # Validation
            val_cost, val_acc, val_duration, _, val_pred, val_labels ,val_learning_rate= evaluate(val_feature, val_adj, val_mask, val_y,
                                                                                placeholders)
            val_f1 = f1_score(val_labels,val_pred, average='binary', pos_label=1)
            val_precision = precision_score(val_labels,val_pred,average='binary', pos_label=1)
            val_recall = recall_score(val_labels,val_pred,average='binary', pos_label=1)

            # Test
            test_cost, test_acc, test_duration, embeddings, pred, labels ,test_learning_rate= evaluate(test_feature, test_adj, test_mask,
                                                                                    test_y,
                                                                                    placeholders)

            test_recall = recall_score(labels,pred,average='binary', pos_label=1)
            test_f1 = f1_score( labels,pred,average='binary', pos_label=1)
            test_precision = precision_score(labels,pred,average='binary', pos_label=1)
            test_mcc = matthews_corrcoef(labels ,pred)

            if epoch > 150:
                if val_f1 > max_f1:
                    max_f1 = val_f1
                    best_epoch = epoch
                    End_test_f1 = test_f1
                    End_test_acc = test_acc
                    End_test_precision = test_precision
                    End_test_recall = test_recall
                    End_test_mcc = test_mcc



            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                  "val_loss=", "{:.5f}".format(val_cost), "val_f1=", "{:.3f}".format(val_f1), "test_pre=",
                  "{:.3f}".format(test_precision), "test_recall=",
                  "{:.3f}".format(test_recall), "test_f1=", "{:.3f}".format(test_f1),
                  "time=", "{:.5f}".format(time.time() - t),
                  "test_acc=","{:.3f}".format(test_acc),
                  "test_mcc=", "{:.3f}".format(test_mcc),
                  "learning_rate=", "{:.8f}".format(lr))


            result = "Epoch:" + "%04d" % (epoch + 1) + " train_loss=" + "{:.5f} ".format(
                train_loss) + " val_loss=" + "{:.5f}".format(val_cost) + " val_f1=" + "{:.3f}".format(
                val_f1) + " test_precision=" + "{:.3f} ".format(
                test_precision) + " test_recall=" + "{:.3f} ".format(
                test_recall) + " test_f1=" + "{:.3f} ".format(
                test_f1) + " time=" + "{:.5f} ".format(time.time() - t) \
                     + " test_acc=" + "{:.3f} ".format(test_acc) \
                     + " test_mcc=" + "{:.3f} ".format(test_mcc) \
                     + " learning_rate=" + "{:.8f} ".format(lr)
            f.write(result)
            f.write('\n')


print("Optimization Finished!")

print("Test F1-Score, Accuracy, Precision, Recall and MCC...")
# Best results
print('epoch:', best_epoch)
print("Test set results:", "f1-score=", "{:.3f}".format(End_test_f1),
      "accuracy=", "{:.3f}".format(End_test_acc),
      "precision=", "{:.3f}".format(End_test_precision),
      "recall=", "{:.3f}".format(End_test_recall),
      "mcc=", "{:.3f}".format(End_test_mcc)
      )
