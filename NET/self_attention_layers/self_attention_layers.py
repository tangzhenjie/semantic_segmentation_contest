import tensorflow as tf
slim = tf.contrib.slim

def position_attention_module(feature, wights=1):
    """
    Position Attention Module
        :param feature: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :return: the shape is same to feature.
    """
    BATCH_SIZE, WIDTH, HEIGHT, DEPTH = feature.get_shape().as_list()
    with tf.variable_scope("position_module"):
        value = slim.conv2d(feature, DEPTH, [1, 1], activation_fn=None, scope="value")
        value = tf.reshape(value, [-1, WIDTH*HEIGHT, DEPTH])
        value = tf.transpose(value, [0, 2, 1])

        query = slim.conv2d(feature, DEPTH, [1, 1], activation_fn=None, scope="query")
        key = slim.conv2d(feature, DEPTH, [1, 1], activation_fn=None, scope="key")
        query = tf.reshape(query, [-1, WIDTH*HEIGHT, DEPTH])
        key = tf.reshape(key, [-1, WIDTH*HEIGHT, DEPTH])
        query = tf.transpose(query, [0, 2, 1])
        mul_end = tf.matmul(key, query)  # shape[batch_size, WIDTH*HEIGHT, WIDTH*HEIGHT]
        s = tf.nn.softmax(mul_end, dim=1)  # shape[batch_size, WIDTH*HEIGHT, WIDTH*HEIGHT]
        s = tf.transpose(s, [0, 2, 1])

        position_ends = tf.matmul(value, s)

        position_ends = tf.transpose(position_ends, [0, 2, 1])
        position_ends = tf.reshape(position_ends, [-1, WIDTH, HEIGHT, DEPTH])
        result = wights * position_ends + feature
        return result
def chanel_attention_module(feature, wights=1):
    """
    Position Attention Module
        :param feature: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :return: the shape is same to feature.
    """
    BATCH_SIZE, WIDTH, HEIGHT, DEPTH = feature.get_shape().as_list()
    with tf.variable_scope("chanel_module"):
        value = tf.reshape(feature, [-1, WIDTH*HEIGHT, DEPTH])
        value = tf.transpose(value, [0, 2, 1])

        query = tf.reshape(feature, [-1, WIDTH*HEIGHT, DEPTH])
        key = tf.reshape(feature, [-1, WIDTH*HEIGHT, DEPTH])
        query = tf.transpose(query, [0, 2, 1])
        mul_end = tf.matmul(query, key)  # shape[batch_size, DEPTH, DEPTH]
        s = tf.nn.softmax(mul_end, dim=1)  # shape[batch_size, DEPTH, DEPTH]
        s = tf.transpose(s, [0, 2, 1])

        position_ends = tf.matmul(s, value) # shape[batch_size, DEPTH, WIDTH*HEIGHT]

        position_ends = tf.transpose(position_ends, [0, 2, 1])
        position_ends = tf.reshape(position_ends, [-1, WIDTH, HEIGHT, DEPTH])
        result = wights * position_ends + feature
        return result
if __name__ == "__main__":
    b = tf.ones(shape=[1, 50, 50, 3])
    x = position_attention_module(b)
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        z = sess.run(x)
        print(z)






