import tensorflow as tf
import numpy as np


class RNN:

    def __init__(self,sequence_length, num_classes, vocab_size, embedding_size,
                 cell_type, hidden_size, l2_reg_lambda):
        '''
        :param sequence_length: 400
        :param num_classes: 2
        :param vocab_size: 25236
        :param embedding_size: 300
        :param cell_type: lstm
        :param hidden_size: 128
        :param l2_reg_lambda: 3.0
        '''

        # placeholder 可以指定shape也可以不用指定，当设定二维的时候必须指定shape，因为只能让其中一个随机。
        # 当设定一维的时候可以不指定shape
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        # dropout，就是指网络中每个单元在每次有数据流入时以一定的概率（keep prob）正常工作，否则输出0值。
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # self.pre_trianing = pre_trianing

        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_text)

        with tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)


        # Recurrent Neural Network
        # bi_lstm 从网上找的模型：https://www.cnblogs.com/Luv-GEM/p/10788849.html
        # all_outputs输出 是从https://blog.csdn.net/zhylhy520/article/details/86364789借鉴到的
        '''with tf.name_scope("bi_lstm"):  # word_embedding
            cell = self._get_cell(hidden_size, cell_type)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.embedded_chars,
                                               sequence_length=text_length,
                                               dtype=tf.float32,)
            self.h_outputs = self.last_relevant(all_outputs, text_length)
            drop = tf.nn.dropout(self.h_outputs , self.dropout_keep_prob)'''
        with tf.name_scope("rnn"):
            cell = self._get_cell(hidden_size, cell_type)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.embedded_chars,
                                               sequence_length=text_length,
                                               dtype=tf.float32)
            self.h_outputs = self.last_relevant(all_outputs, text_length)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            # W = tf.get_variable("W", shape=[2*hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            # tf.argmax:根据axis取值的不同返回每行或者每列最大值的索引。
            # axis = 0时比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组
            # axis = 1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



    # Length of the sequence data
    @staticmethod
    def _length(seq):
        # tf.abs(seq) 计算张量的绝对值，可以将多个数值传入list，统一求绝对值
        # 如果x < 0,则有 y = sign(x) = -1；如果x == 0,则有 0 或者tf.is_nan(x)；如果x > 0,则有1.
        relevant = tf.sign(tf.abs(seq))
        # tf.reduce_sum求和函数
        length = tf.reduce_sum(relevant, axis=1)
        # tf.cast 可以将x的数据格式转化成dtype
        length = tf.cast(length, tf.int32)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.



    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])

        return tf.gather(flat, index)