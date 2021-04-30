import tensorflow as tf
import numpy as np

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

    # dice loss
    # intersection = tf.reduce_sum(labels*preds,axis=1)
    # union = tf.reduce_sum(labels,axis=1) + tf.reduce_sum(preds,axis=1) + 1e-5
    # dice = 1. - (2*intersection/union)
    # loss = 1-tf.reduce_mean(dice)
    # return loss

# GHM  https://github.com/GXYM/GHM_Loss
def ghm_class_loss(logits, targets, masks=None):
    """ Args:
    input [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, class_num]:
        Binary target (0 or 1) for each sample each class. The value is -1
        when the sample is ignored.
    """
    train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
    g_v = tf.abs(tf.sigmoid(logits) - targets)  # [batch_num, class_num]
    g = tf.expand_dims(g_v, axis=0)  # [1, batch_num, class_num]

    if masks is None:
        masks = tf.ones_like(targets)
    valid_mask = masks > 0
    weights, tot = calc(g, valid_mask)
    print(weights.shape)
    ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets * train_mask,
                                                             logits=logits)
    ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

    return ghm_class_loss
def calc(g, valid_mask):
    bins=10
    momentum = 0
    edges_left = [float(x) / bins for x in range(bins)]
    edges_left = tf.constant(edges_left)  # [bins]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

    edges_right = [float(x) / bins for x in range(1, bins + 1)]
    edges_right[-1] += 1e-3
    edges_right = tf.constant(edges_right)  # [bins]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
    alpha = momentum
    # valid_mask = tf.cast(valid_mask, dtype=tf.bool)
    tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
    inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
    zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
    valid_bins = tf.greater(num_in_bin, 0)  # [bins]

    num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

    if momentum > 0:
        acc_sum = [0.0 for _ in range(bins)]
        acc_sum = tf.Variable(acc_sum, trainable=False)

    if alpha > 0:
        update = tf.assign(acc_sum,
                           tf.where(valid_bins, alpha * acc_sum + (1 - alpha) * num_in_bin, acc_sum))
        with tf.control_dependencies([update]):
            acc_sum_tmp = tf.identity(acc_sum, name='updated_accsum')
            acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
            acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
    else:
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
        num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
        weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
        weights = tf.reduce_sum(weights, axis=0)
    weights = weights / num_valid_bin
    return weights, tot



def dice_loss(preds, labels,smooth=1): #   DL
    inse = tf.reduce_sum(preds * labels,axis=1)
    l =  tf.reduce_sum(preds * preds,axis=1)
    r = tf.reduce_sum(labels * labels,axis=1)
    dice = (2. * inse + smooth) / (l + r + smooth)
    loss = tf.reduce_mean(1-dice)
    return loss

def focal_loss_fixed(y_pred,y_true):

    epsilon = 1.e-9
    gamma = 2.
    alpha = 0.25
    y_pred = tf.nn.sigmoid(y_pred)
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    model_out = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # to advoid numeric underflow

    # compute cross entropy ce = ce_0 + ce_1 = - (1-y)*log(1-y_hat) - y*log(y_hat)
    ce_0 = tf.multiply(tf.subtract(1., y_true), -tf.log(tf.subtract(1., model_out)))
    ce_1 = tf.multiply(y_true, -tf.log(model_out))

    # compute focal loss fl = fl_0 + fl_1
    # obviously fl < ce because of the down-weighting, we can fix it by rescaling
    # fl_0 = -(1-y_true)*(1-alpha)*((y_hat)^gamma)*log(1-y_hat) = (1-alpha)*((y_hat)^gamma)*ce_0
    fl_0 = tf.multiply(tf.pow(model_out, gamma), ce_0)
    fl_0 = tf.multiply(1. - alpha, fl_0)
    # fl_1= -y_true*alpha*((1-y_hat)^gamma)*log(y_hat) = alpha*((1-y_hat)^gamma*ce_1
    fl_1 = tf.multiply(tf.pow(tf.subtract(1., model_out), gamma), ce_1)
    fl_1 = tf.multiply(alpha, fl_1)
    fl = tf.add(fl_0, fl_1)
    f1_avg = tf.reduce_mean(fl)
    return f1_avg



# def focal_loss(logits, labels,samples_per_cls):
#     '''
#     :param logits:  [batch_size, n_class]
#     :param labels: [batch_size]
#     :return: -(1-y)^r * log(y)
#     gamma=2
#     '''
#     no_of_classes = 2
#     beta = 0.9999
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes
#     weights = tf.reduce_sum(weights * labels, axis=1)
#     unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
#     unweighted_loss = tf.reduce_sum(unweighted_loss,axis=1)
#     unweighted_loss /= 2
#     # unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
#     weights_loss = unweighted_loss*weights
#     focal_loss = tf.reduce_mean(weights_loss)
#     return focal_loss

# def tversky_loss(y_pred, y_true): # https://www.cnblogs.com/hotsnow/p/10954624.html
#     y_pred = tf.nn.softmax(y_pred)
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#     alpha = 0.7
#     # alpha = 0.3 #当alpha=0.5 的时候就是DSC方法
#     smooth = 1
#     pt_1 = ((true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth))
#     gamma = 0.75
#     return K.pow((1-pt_1),gamma)

def xiaxin_weighted_cross_entropy(preds,labels,samples_per_cls): # 2020 7 24
    d_lossweight =samples_per_cls[0]/samples_per_cls[1] # 跟每个数据集标签的变化而变化
    print(" d_lossweight", d_lossweight)
    class_weights = tf.constant([1.0,d_lossweight])
    weights = tf.reduce_sum(class_weights * labels, axis=1)
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    weighted_losses = unweighted_loss * weights
    loss = tf.reduce_mean(weighted_losses)
    return loss

# RNN 模型loss
def weighted_cross_entropy(preds, labels):
    loss = tf.nn.weighted_cross_entropy_with_logits(logits = preds,targets=labels,pos_weight=0.25)
    # l2_reg_lambda=0.2 beta=0.5
    # return tf.reduce_mean(loss)
    return tf.reduce_mean(loss)

def focal_loss_fixed_new(y_pred,y_true):
    gamma = 2
    alpha = 0.55
    epsilon = 1.e-9
    y_pred = tf.nn.softmax(y_pred)
    y_true = tf.convert_to_tensor(y_true,tf.float32)
    y_pred = tf.convert_to_tensor(y_pred,tf.float32)
    model_out = tf.add(y_pred,epsilon)
    ce = tf.multiply(y_true,-tf.log(model_out))
    weight = tf.multiply(y_true,tf.pow(tf.subtract(1.,model_out),gamma))
    f1 = tf.multiply(alpha,tf.multiply(weight,ce))
    reduced_f1 = tf.reduce_max(f1,axis=1)
    return tf.reduce_mean(reduced_f1)



def DSC_loss(y_pred,y_true): #https://www.cnblogs.com/CheeseZH/p/13554252.html
    epsilon = 1.e-6
    alpha = 2.0
    smooth = 1.e-5
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_pred_mask = tf.multiply(y_pred, y_true)
    common = tf.multiply((tf.ones_like(y_true) - y_pred_mask), y_pred_mask)
    nominator = tf.multiply(tf.multiply(common, y_true), alpha) + smooth
    denominator = common + y_true + smooth
    dice_coe = tf.divide(nominator, denominator)
    return tf.reduce_mean(tf.reduce_max(1 - dice_coe, axis=-1))

def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)
