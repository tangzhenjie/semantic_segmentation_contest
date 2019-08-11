import numpy as np
import cv2
from tqdm import tqdm
import random
import tensorflow as tf
import sys
from libtiff import TIFF
from scipy import io
img_w = 1000
img_h = 1000
train_sets = ['train/1', 'train/2', 'train/3', 'train/4',
              'train/5', 'train/6', 'train/7', 'train/8']
val_sets = ['val/1', 'val/2']

class0 = np.array([0, 0, 0])
class1 = np.array([0, 200, 0])
class2 = np.array([150, 250, 0])
class3 = np.array([150, 200, 150])
class4 = np.array([200, 0, 200])
class5 = np.array([150, 0, 250])
class6 = np.array([150, 150, 250])
class7 = np.array([250, 200, 0])
class8 = np.array([200, 200, 0])
class9 = np.array([200, 0, 0])
class10 = np.array([250, 0, 150])
class11 = np.array([200, 150, 150])
class12 = np.array([250, 150, 150])
class13 = np.array([0, 0, 200])
class14 = np.array([0, 150, 200])
class15 = np.array([0, 200, 250])

def creat_dataset(image_num=100000, image_sets=train_sets, type='train', mode='original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        tif = TIFF.open('../DatasetOrigin/' + image_sets[i] + '.tif', mode='r')  # 4 channels
        src_img = tif.read_image()
        #src_img_new = cv2.imread('../DatasetOrigin/' + image_sets[i] + '.tif', cv2.IMREAD_UNCHANGED)  # 4 channels
        #label_img = cv2.imread('../DatasetOrigin/' + image_sets[i] + '_label.tif', cv2.IMREAD_COLOR)  # 3 channels
        tif_label = TIFF.open('../DatasetOrigin/' + image_sets[i] + '_label.tif', mode='r')
        label_img = tif_label.read_image()
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

            #visualize = np.zeros((256, 256)).astype(np.uint8)
            #visualize = label_roi * 50

            if type == "train":
                #cv2.imwrite(('../DatasetNew/train/images/%d.tif' % g_count), src_roi)
                io.savemat('../DatasetNew/train/images/%d.mat' % g_count, {"feature": src_roi})
                #cv2.imwrite(('../DatasetNew/train/labels/%d.png' % g_count), label_roi)
                io.savemat('../DatasetNew/train/labels/%d.mat' % g_count, {"feature": label_roi})
            else:
                #cv2.imwrite(('../DatasetNew/val/images/%d.tif' % g_count), src_roi)
                io.savemat('../DatasetNew/val/images/%d.mat' % g_count, {"feature": src_roi})
                #cv2.imwrite(('../DatasetNew/val/labels/%d.png' % g_count), label_roi)
                io.savemat('../DatasetNew/val/labels/%d.mat' % g_count, {"feature": label_roi})
            count += 1
            g_count += 1
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _datas_to_tfexample(data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': _bytes_feature(data),
        'label': _bytes_feature(label)
        }))


def to_tfrecord_train(train_filename, length):
    tfrecord_writer = tf.python_io.TFRecordWriter(train_filename)
    for key in tqdm(range(length)):
        #img_data = cv2.imread('../DatasetNew/train/images/%d.tif' % key, cv2.IMREAD_UNCHANGED)
        img_data = io.loadmat('../DatasetNew/train/images/%d.mat' % key)["feature"]  # uint8
        label_data = cv2.imread('../DatasetNew/train/labels_2d/%d.png' % key, cv2.IMREAD_GRAYSCALE)  # int8
        img_data = img_data.tobytes()
        label_data = label_data.tobytes()
        # 生成example
        example = _datas_to_tfexample(img_data, label_data)
        tfrecord_writer.write(example.SerializeToString())
        #sys.stdout.write("\r>> Converting image" + output_filename)
    tfrecord_writer.close()
    sys.stdout.flush()
def to_tfrecord_val(val_filename, length):
    tfrecord_writer = tf.python_io.TFRecordWriter(val_filename)
    for key in tqdm(range(length)):
        #img_data = cv2.imread('../DatasetNew/val/images/%d.tif' % key, cv2.IMREAD_UNCHANGED)
        img_data = io.loadmat('../DatasetNew/val/images/%d.mat' % key)["feature"]
        label_data = cv2.imread('../DatasetNew/val/labels_2d/%d.png' % key, cv2.IMREAD_GRAYSCALE)
        img_data = img_data.tobytes()
        label_data = label_data.tobytes()
        # 生成example
        example = _datas_to_tfexample(img_data, label_data)
        tfrecord_writer.write(example.SerializeToString())
        #sys.stdout.write("\r>> Converting image" + output_filename)
    tfrecord_writer.close()
    sys.stdout.flush()

def encode_labels(image_color_data):
    iamge_data_RGB = image_color_data
    height, width, chanel = iamge_data_RGB.shape
    label_seg = np.zeros([height, width], dtype=np.int8)
    label_seg[(iamge_data_RGB == class0).all(axis=2)] = 0
    label_seg[(iamge_data_RGB == class1).all(axis=2)] = 1
    label_seg[(iamge_data_RGB == class2).all(axis=2)] = 2
    label_seg[(iamge_data_RGB == class3).all(axis=2)] = 3
    label_seg[(iamge_data_RGB == class4).all(axis=2)] = 4
    label_seg[(iamge_data_RGB == class5).all(axis=2)] = 5
    label_seg[(iamge_data_RGB == class6).all(axis=2)] = 6
    label_seg[(iamge_data_RGB == class7).all(axis=2)] = 7
    label_seg[(iamge_data_RGB == class8).all(axis=2)] = 8
    label_seg[(iamge_data_RGB == class9).all(axis=2)] = 9
    label_seg[(iamge_data_RGB == class10).all(axis=2)] = 10
    label_seg[(iamge_data_RGB == class11).all(axis=2)] = 11
    label_seg[(iamge_data_RGB == class12).all(axis=2)] = 12
    label_seg[(iamge_data_RGB == class13).all(axis=2)] = 13
    label_seg[(iamge_data_RGB == class14).all(axis=2)] = 14
    label_seg[(iamge_data_RGB == class15).all(axis=2)] = 15
    return label_seg


if __name__ == "__main__":
    creat_dataset(image_num=500, image_sets=val_sets, type='val', mode='original')
    creat_dataset(image_num=5000, image_sets=train_sets, type='train', mode='original')

    # 把RGB标签换成灰度标签
    for key in tqdm(range(5000)):
        src_img = io.loadmat('../DatasetNew/train/labels/%d.mat' % key)
        new_data = encode_labels(src_img["feature"])
        result = cv2.imwrite("../DatasetNew/train/labels_2d/%d.png" % key, new_data)
    for key in tqdm(range(500)):
        src_img = io.loadmat('../DatasetNew/val/labels/%d.mat' % key)
        new_data = encode_labels(src_img["feature"])
        result = cv2.imwrite('../DatasetNew/val/labels_2d/%d.png' % key, new_data)
    to_tfrecord_train(train_filename="../DatasetNew/train/train3.tfrecord", length=5000)
    to_tfrecord_val(val_filename="../DatasetNew/val/val3.tfrecord", length=500)