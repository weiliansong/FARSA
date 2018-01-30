import definition
import tensorflow as tf
slim = tf.contrib.slim

spec = definition.spec
num_classes = definition.get_num_classes()

def vgg_arg_scope():
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg16(inputs):
  with tf.variable_scope('vgg_16', 'vgg_16', [inputs]) as sc:
    with slim.arg_scope(vgg_arg_scope()):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      return net

def build(images, batch_size):
  pool5 = vgg16(images)

  with tf.variable_scope('common_final'):
    with slim.arg_scope(vgg_arg_scope()):
      common = slim.fully_connected(pool5, 512)
      common = tf.reshape(common, (-1, 210, 512))

  logits = {}

  with tf.variable_scope('sr_final'):
    with slim.arg_scope(vgg_arg_scope()):
      print('Softmax attention mechanism for SR')
      attention = tf.get_variable('attention', (1,210), 
                          initializer=tf.contrib.layers.xavier_initializer())
      attention = tf.nn.softmax(attention)
      batch_attention = tf.reshape(tf.tile(attention, 
                                          (batch_size,1)), 
                                          (batch_size,1,210))
      sr_net = tf.matmul(batch_attention, common)
      sr_net = tf.contrib.layers.flatten(sr_net)

      logits['sr'] = slim.fully_connected(sr_net, 
                                          num_classes['sr'],
                                          activation_fn=None)
  for key in spec.keys():
    with tf.variable_scope('%s_final' % key):
      with slim.arg_scope(vgg_arg_scope()):
        attention = tf.get_variable('attention', (1,210), 
                            initializer=tf.contrib.layers.xavier_initializer())
        attention = tf.nn.softmax(attention)
        batch_attention = tf.reshape(tf.tile(attention, 
                                            (batch_size,1)), 
                                            (batch_size,1,210))
        net = tf.matmul(batch_attention, common)
        net = tf.contrib.layers.flatten(net)

        logits[key] = slim.fully_connected(net, 
                                           num_classes[key],
                                           activation_fn=None)
  return logits
