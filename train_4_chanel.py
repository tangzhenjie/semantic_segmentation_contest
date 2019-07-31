from DataGenerate.GetDataset import train_or_eval_input_fn
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import os
import argparse
import tools
import datetime
import math
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 4
summary_path = "./summary_4/"
checkpoint_path_first = "/2T/tzj/deeplabv3/checkpoint/"
checkpoint_path = "./checkpoint_4/"
EPOCHS = 100
train_set_length = 5000
eval_set_length = 500

parser = argparse.ArgumentParser()

#添加参数
envarg = parser.add_argument_group('Training params')
# BN params
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument('--freeze_batch_norm', type=bool, default=False,  help='Freeze batch normalization parameters during the training.')
# the number of classes
envarg.add_argument("--number_of_classes", type=int, default=16, help="Number of classes to be predicted.")

# regularizer
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")

# for deeplabv3
envarg.add_argument("--multi_grid", type=list, default=[1, 2, 4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")

# the base network
envarg.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")

# the pre_trained model for example resnet50 101 and so on
envarg.add_argument('--pre_trained_model', type=str, default='./pre_trained_model/resnet_v2_50/resnet_v2_50.ckpt',
                    help='Path to the pre-trained model checkpoint.')

# max number of batch elements to tensorboard
parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')
# poly learn_rate
parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')
parser.add_argument('--max_iter', type=int, default=125000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')
args = parser.parse_args()

def main():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'

    is_train = tf.placeholder(tf.bool, shape=[])
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1000, 1000, 4], name="image_batch")
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1000, 1000, 1], name="label_batch")

    train_dataset = train_or_eval_input_fn(is_training=True,
                                           data_dir="/2T/tzj/semantic_segmentation_contest/DatasetNew/train/", batch_size=batch_size)
    eval_dataset = train_or_eval_input_fn(is_training=False,
                                           data_dir="/2T/tzj/semantic_segmentation_contest/DatasetNew/val/", batch_size=batch_size, num_epochs=1)
    iterator_train = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_batch = iterator_train.get_next()
    training_init_op = iterator_train.make_initializer(train_dataset)
    evaling_init_op = iterator_train.make_initializer(eval_dataset)

    loss, train_op, predictions, metrics = tools.get_loss_pre_metrics(x, y, is_train, batch_size, args)

    accuracy = metrics["px_accuracy"]
    mean_iou = metrics["mean_iou"]
    confusion_matrix = predictions['confusion_matrix']

    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    )
    # 首次运行从deeplabv3中获取权重需要剔除logits层
    exclude = [args.resnet_model + '/logits', args.resnet_model + '/conv1', 'DeepLab_v3/logits', 'global_step']
    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    with tf.variable_scope("resnet_v2_50", reuse=True):
        conv1_weight_restored = tf.get_variable("conv1/weights")
        conv1_biase_restored = tf.get_variable("conv1/biases")
    # 获得变量值
    checkpoint_path_restored = os.path.join(checkpoint_path_first, "model.ckpt-1")
    reader_restored = pywrap_tensorflow.NewCheckpointReader(checkpoint_path_restored)
    conv1_weight_value = reader_restored.get_tensor("resnet_v2_50/conv1/weights")
    conv1_weight_value_4 = np.sum(conv1_weight_value, axis=2, keepdims=True)
    conv1_weight_value_4 = np.true_divide(conv1_weight_value_4, 3)
    conv1_weight_value = np.concatenate((conv1_weight_value_4, conv1_weight_value), axis=2)
    conv1_biase_value = reader_restored.get_tensor("resnet_v2_50/conv1/biases")
    conv1_weight_op = tf.assign(conv1_weight_restored, conv1_weight_value)
    conv1_biase_op = tf.assign(conv1_biase_restored, conv1_biase_value)


    saver_first = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=100)
    summary_writer_train = tf.summary.FileWriter(summary_path + "train/")
    summary_writer_val = tf.summary.FileWriter(summary_path + "val/")
    # 运行图
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op, feed_dict={is_train: True})
        ckpt = tf.train.get_checkpoint_state(checkpoint_path_first)
        if ckpt and ckpt.model_checkpoint_path:
            saver_first.restore(sess, ckpt.model_checkpoint_path)
        sess.graph.finalize()
        sess.run([conv1_weight_op, conv1_biase_op])

        train_batches_of_epoch = int(math.ceil(train_set_length / batch_size))
        val_batches_of_epoch = int(math.ceil(eval_set_length / batch_size))
        for epoch in range(EPOCHS):
            sess.run(training_init_op)
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
            # step = 1
            for step in range((epoch * train_batches_of_epoch), ((epoch + 1) * train_batches_of_epoch)):
                img_batch, label_batch = sess.run(next_batch)
                loss_value, _, acc, m_iou, con_matrix = sess.run(
                    [loss, train_op, accuracy, mean_iou, confusion_matrix],
                    feed_dict={x: img_batch, y: label_batch, is_train: True})

                if (step + 1) % 625 == 0:
                    kappa = tools.kappa(con_matrix)
                    print("{} {} loss = {:.4f}".format(datetime.datetime.now(), step + 1, loss_value))
                    print("accuracy{}".format(acc))
                    print("miou{}".format(m_iou))
                    print("kappa{}".format(kappa))
                    merge = sess.run(summary_op, feed_dict={x: img_batch, y: label_batch, is_train: True})
                    summary_writer_train.add_summary(merge, step + 1)
            saver.save(sess, checkpoint_path + "model.ckpt", epoch + 1)
            print("checkpoint saved")

            # 验证过程
            sess.run(evaling_init_op)
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.0
            test_miou = 0.0
            test_kappa = 0.0
            test_count = 0
            for tag in range(val_batches_of_epoch):
                img_batch, label_batch = sess.run(next_batch)
                acc, m_iou, con_matrix = sess.run(
                    [accuracy, mean_iou, confusion_matrix],
                    feed_dict={x: img_batch, y: label_batch, is_train: False})

                kappa = tools.kappa(con_matrix)
                test_kappa += kappa
                test_acc += acc
                test_miou += m_iou
                test_count += 1
            test_acc /= test_count
            test_miou /= test_count
            test_kappa /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc),
                tf.Summary.Value(tag="validation_miou", simple_value=test_miou),
                tf.Summary.Value(tag="validation_kappa", simple_value=test_kappa)
            ])
            summary_writer_val.add_summary(s, epoch + 1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
            print("{} Validation miou = {:.4f}".format(datetime.datetime.now(), test_miou))
            print("{} Validation kappa = {:.4f}".format(datetime.datetime.now(), test_kappa))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()