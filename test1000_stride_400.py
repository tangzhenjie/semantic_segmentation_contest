import numpy as np
import cv2
import os
import argparse
import tensorflow as tf
from NET import deeplab_v3
from utils.preprocessing import decode_labels

checkpoint_path = "./checkpoint_1000/"
image_size = 1000
stride = 400
def predict(TEST_SET, sess, prediction, imgs_batch):
    for n in range(len(TEST_SET)):

        path = TEST_SET[n]  # load the image
        image = cv2.imread('/2T/tzj/semantic_segmentation_contest/DatasetNew/test/' + path, cv2.IMREAD_UNCHANGED)
        h, w, chanel = image.shape

        result = np.zeros((h, w, 3), dtype=np.uint8)

        padding_h = h + 600
        padding_w = w + 600
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[300:300 + h, 300:300 + w, :] = image[:, :, 0:3]
        mask_whole = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        for i in range(((padding_h - 1000) // stride) + 1):
            for j in range(((padding_w - 1000) // stride) + 1):
                crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :]
                ch, cw, _ = crop.shape
                if ch != 1000 or cw != 1000:
                    print('invalid size!')
                    continue
                img_batch = np.expand_dims(crop, axis=0)
                pred = sess.run(prediction, feed_dict={imgs_batch: img_batch})
                pred = decode_labels(pred)
                mask_whole[300 + i * stride: 300 + i * stride + stride, 300 + j * stride: 300 + j * stride + stride, :] = pred[0][300:700, 300:700, :]
                print("i:%d" % i, "j:%d" % j)
        result[0:h, 0:w, :] = mask_whole[300:300 + h, 300:300 + w, :]
        result_bgr = result[..., ::-1]
        cv2.imwrite("./Result1000_stride_400/" + path.split(".")[0] + "_label.tif", result_bgr)
        print("./Result_stride_400/" + path.split(".")[0] + "_label.tif  saved")

#cv2.imwrite('./predict/pre' + str(n + 1) + '.png', mask_whole[0:h, 0:w])
# 获取全部测试集图片
TEST_SET = os.listdir('/2T/tzj/semantic_segmentation_contest/DatasetNew/test')

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
parser.add_argument('--initial_learning_rate', type=float, default=7e-4,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')
parser.add_argument('--max_iter', type=int, default=25000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')
args = parser.parse_args()

img_batch = tf.placeholder("float32", shape=[None, 1000, 1000, 3], name="img_batch")
# 损失

logits = deeplab_v3.deeplab_v3(img_batch, args, is_training=False, reuse=False)
prediction = tf.argmax(logits, axis=3)
prediction = tf.expand_dims(prediction, axis=3)

init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    )
saver = tf.train.Saver()
# 运行图
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # 恢复权重
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, checkpoint_path + "model.ckpt-26")
        print("restored!!!")
    predict(TEST_SET=TEST_SET, sess=sess, prediction=prediction, imgs_batch=img_batch)