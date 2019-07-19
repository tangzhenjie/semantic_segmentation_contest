from utils import dataset_util
from DataGenerate.GetDataset import eval_or_test_input_fn
import tensorflow as tf
import os
import argparse

evaluation_data_list = "./VOC2012_AUG/txt/val.txt"
parser = argparse.ArgumentParser()

#添加参数
envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=16, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.00001, help="initial learning rate.")
envarg.add_argument('--learning_rate', type=float, default=0.00001, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1, 2, 4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")
envarg.add_argument("--crop_size", type=int, default=513, help="Image Cropsize.")
envarg.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")
envarg.add_argument('--learning_power', type=float, default=0.9, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")
args = parser.parse_args()
def main():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'

    # 获取数据
    examples = dataset_util.read_examples_list(FLAGS.evaluation_data_list)
    image_files = [os.path.join(FLAGS.image_data_dir, filename) + '.jpg' for filename in examples]
    label_files = [os.path.join(FLAGS.label_data_dir, filename) + '.png' for filename in examples]

    features, labels = eval_or_test_input_fn.eval_input_fn(image_files, label_files)

    # Manually load the latest checkpoint
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)