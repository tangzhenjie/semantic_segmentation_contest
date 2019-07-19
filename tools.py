from NET import network
from utils import preprocessing
import tensorflow as tf

_WEIGHT_DECAY = 5e-4


def get_loss_pre_metrics(x, y, is_training, batch_size, args):
    # 恢复图像
    images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, x),
      tf.uint8)

    # 前向传播
    logits = tf.cond(is_training, true_fn=lambda: network.deeplab_v3(x, args, is_training=True, reuse=False),
                     false_fn=lambda: network.deeplab_v3(x, args, is_training=False, reuse=True))
    pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

    # 解码预测结果
    pred_decoded_labels = tf.cond(is_training, true_fn=lambda: tf.py_func(preprocessing.decode_labels, [pred_classes, batch_size, args.number_of_classes], tf.uint8),
            false_fn=lambda: tf.py_func(preprocessing.decode_labels, [pred_classes, 1, args.number_of_classes], tf.uint8))


    predictions = {
        'pred': pred_classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'decoded_labels': pred_decoded_labels
    }

    # 解码标签
    gt_decoded_labels = tf.cond(is_training,
            true_fn=lambda: tf.py_func(preprocessing.decode_labels, [y, batch_size, args.number_of_classes], tf.uint8),
            false_fn=lambda: tf.py_func(preprocessing.decode_labels, [y, 1, args.number_of_classes], tf.uint8))

    tf.summary.image('images', tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=args.tensorboard_images_max_outputs)

    # 求loss
    labels = tf.squeeze(y, axis=3)  # reduce the channel dimension.
    logits_by_num_classes = tf.reshape(logits, [-1, args.number_of_classes])
    labels_flat = tf.reshape(labels, [-1, ])

    # 去掉没有标签的255
    valid_indices = tf.to_int32(labels_flat <= args.number_of_classes - 1)
    logits_by_num_classes_new = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    labels_flat_new = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]


    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits_by_num_classes_new, labels=labels_flat_new)

    if not args.freeze_batch_norm:
        train_var_list = [v for v in tf.trainable_variables()]
    else:
        train_var_list = [v for v in tf.trainable_variables()
                          if 'beta' not in v.name and 'gamma' not in v.name]

    #train_var_list = [v for v in tf.trainable_variables()]
    with tf.variable_scope("total_loss"):
        loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])
    tf.summary.scalar('loss', loss)

    # 优化函数
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(
        args.initial_learning_rate,
        tf.cast(global_step, tf.int32) - args.initial_global_step,
        args.max_iter, args.end_learning_rate, power=0.9)  # args.max_iter = 30000 args.initial_global_step=0
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-6, momentum=0.9)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)

    # metrics
    preds_flat = tf.reshape(pred_classes, [-1, ])
    preds_flat_new = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
    confusion_matrix = tf.confusion_matrix(labels_flat_new, preds_flat_new, num_classes=args.number_of_classes)

    predictions['confusion_matrix'] = confusion_matrix

    correct_pred = tf.equal(preds_flat_new, labels_flat_new)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    def compute_mean_iou(total_cm, name='mean_iou'):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(
            tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0),
            denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        for i in range(args.number_of_classes):
            tf.identity(iou[i], name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name=name) / num_valid_entries,
            0)
        return result

    mean_iou = compute_mean_iou(confusion_matrix)

    tf.summary.scalar('mean_iou', mean_iou)

    metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

    return loss, train_op, predictions, metrics