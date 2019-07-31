import tensorflow as tf
from NET.resnet_v2_psp import resnet_utils, resnet_v2

slim = tf.contrib.slim
def psp_conv(x, kernel_size, scope_name, is_training=True):
    filters_in = x.get_shape()[-1]
    with tf.variable_scope(scope_name) as scope:
        kernel = tf.get_variable(
            name='weights',
            shape=[kernel_size, kernel_size, filters_in, 512],
            trainable=True)
    conv_out = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    bn = slim.batch_norm(conv_out, activation_fn=tf.nn.relu, is_training=is_training, scope=scope_name + "/bn")
    return bn

def _pspnet_builder(x,
                    is_training,
                    args,
                    reuse=False):
    """Helper function to build PSPNet model for semantic segmentation.

    The PSPNet model is composed of one base network (ResNet101) and
    one pyramid spatial pooling (PSP) module, followed with concatenation
    and two more convlutional layers for segmentation prediction.

    Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    cnn_fn: A function which builds the base network (ResNet101).
    num_classes: Number of predicted classes for classification tasks.
    is_training: If the tensorflow variables defined in this network
      would be used for training.
    reuse: enable/disable reuse for reusing tensorflow variables. It is
      useful for sharing weight parameters across two identical networks.

    Returns:
    A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
    """
    # Ensure that the size of input data is valid (should be multiple of 6x8=48).
    h, w = x.get_shape().as_list()[1:3] # NxHxWxC
    assert(h == w)
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        cnn_fn = getattr(resnet_v2, args.resnet_model)
        # Build the base network.
        _, end_points = cnn_fn(inputs=x,
                       is_training=is_training,
                       global_pool=False,
                       output_stride=8,
                       spatial_squeeze=False,
                       reuse=reuse)

        if is_training:
            exclude = [args.resnet_model + '/logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(args.pre_trained_model,
                                          {v.name.split(':')[0]: v for v in variables_to_restore})


        x = end_points[args.resnet_model + "/block4"]

        with tf.variable_scope(args.resnet_model, reuse=reuse) as scope:
            # Build the PSP module
            pool_k = int(h/8) # the base network is stride 8 by default.

            # Build pooling layer results in 1x1 output.
            pool1 = tf.nn.avg_pool(x,
                                   name='block5/pool1',
                                   ksize=[1, pool_k, pool_k, 1],
                                   strides=[1, pool_k, pool_k, 1],
                                   padding='VALID')
            pool1 = psp_conv(pool1, 1, 'block5/pool1/conv1', is_training)

            pool1 = tf.image.resize_bilinear(pool1, [pool_k, pool_k])

            # Build pooling layer results in 2x2 output.
            pool2 = tf.nn.avg_pool(x,
                                   name='block5/pool2',
                                   ksize=[1, pool_k//2, pool_k//2, 1],
                                   strides=[1, pool_k//2, pool_k//2, 1],
                                   padding='VALID')
            pool2 = psp_conv(pool2, 1, 'block5/pool2/conv1', is_training)

            pool2 = tf.image.resize_bilinear(pool2, [pool_k, pool_k])

            # Build pooling layer results in 3x3 output.
            pool3 = tf.nn.avg_pool(x,
                                   name='block5/pool3',
                                   ksize=[1, pool_k//3, pool_k//3, 1],
                                   strides=[1, pool_k//3, pool_k//3, 1],
                                   padding='VALID')
            pool3 = psp_conv(pool3, 1, 'block5/pool3/conv1', is_training)

            pool3 = tf.image.resize_bilinear(pool3, [pool_k, pool_k])

            # Build pooling layer results in 6x6 output.
            pool6 = tf.nn.avg_pool(x,
                                   name='block5/pool6',
                                   ksize=[1, pool_k//6, pool_k//6, 1],
                                   strides=[1, pool_k//6, pool_k//6, 1],
                                   padding='VALID')
            pool6 = psp_conv(pool6, 1, 'block5/pool6/conv1', is_training)

            pool6 = tf.image.resize_bilinear(pool6, [pool_k, pool_k])

            # Fuse the pooled feature maps with its input, and generate
            # segmentation prediction.
            x = tf.concat([pool1, pool2, pool3, pool6, x],
                          name='block5/concat',
                          axis=3)
            x = psp_conv(x, 3, 'block5/conv2', is_training)

            x = slim.conv2d(x, args.number_of_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='block5/fc1_voc12', padding='SAME')

            x = tf.image.resize_bilinear(x, [h, w])

            return x

def pspnet_resnet(x, args, is_training, reuse=False):
    """Helper function to build PSPNet model for semantic segmentation.

      The PSPNet model is composed of one base network (ResNet101) and
      one pyramid spatial pooling (PSP) module, followed with concatenation
      and two more convlutional layers for segmentation prediction.

      Args:
        x: A tensor of size [batch_size, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks.
        is_training: If the tensorflow variables defined in this network
          would be used for training.
        use_global_status: enable/disable use_global_status for batch
          normalization. If True, moving mean and moving variance are updated
          by exponential decay.
        reuse: enable/disable reuse for reusing tensorflow variables. It is
          useful for sharing weight parameters across two identical networks.

      Returns:
        A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
      """
    with tf.name_scope('psp') as scope:
        result = _pspnet_builder(x,
                                is_training,
                                args,
                                reuse=reuse)
    return result