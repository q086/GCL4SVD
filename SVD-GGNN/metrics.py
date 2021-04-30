import tensorflow as tf
from tensorflow.python.ops import array_ops


def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

def focal_loss(pred,y):
    gamma = 2
    alpha = 0.25
    pred = tf.nn.softmax(pred)
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = array_ops.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
    return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent,axis=1))