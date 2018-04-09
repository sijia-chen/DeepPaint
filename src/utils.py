import tensorflow as tf
import numpy as np

def linear(input, output_size, scope=None, stddev=0.02, bias_init=0.0):
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input, matrix) + bias

def lrelu(input, alpha = 0.2):
    return tf.maximum(input * alpha, input)

def minibatches(inputs, batch_size, shuffle = True):
    arr_inputs = [np.array(i) for i in inputs]
    indices = np.arange(len(arr_inputs[0]))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(arr_inputs[0]) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield [i[excerpt] for i in arr_inputs]

def log(filepath, content, mode = 'a+', newline = True):
    with open(filepath, mode) as f:
        if newline:
            content += '\n'
        f.write(content)
