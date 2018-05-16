import model
import argparse
import definition
import numpy as np
import tensorflow as tf
from scipy.ndimage import imread

img_means = definition.img_means
spec = definition.spec

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', dest='img_path',
                    action='store', type=str, default='./pano.jpg')
args = parser.parse_args()

# Image Preprocessing
print('Preprocessing Image')
img_data = tf.read_file(args.img_path)
img_data = tf.image.decode_jpeg(img_data, channels=3)
img_data = img_data[167:1330, :, :]
img_data = tf.cast(img_data, tf.float32)
img_data = tf.image.resize_images(img_data, tf.constant([224,960]))

num_channels = img_data.get_shape().as_list()[-1]
channels = tf.split(axis=2, num_or_size_splits=num_channels, value=img_data)
for i in range(num_channels):
  channels[i] -= img_means[i]
img_data = tf.concat(values=channels, axis=2)
img_data = tf.expand_dims(img_data, axis=0)

# Build model
print('Building Model')
with tf.variable_scope('Network') as scope:
  logits = model.build(img_data, batch_size=1)

with tf.Session() as sess:
  print('Session Started')
  sess.run(tf.global_variables_initializer())

  # Restoring from best checkpoint
  print('Restoring from checkpoint')
  restorer = tf.train.Saver()
  restorer.restore(sess, 'checkpoint/farsa.ckpt')

  _logits = sess.run(logits)

  # Star Rating Prediction
  sr_prediction = np.argmax(_logits['sr']) + 1
  print('\nStar Rating Prediction: ' + str(sr_prediction) + ' Star')
  print('')

  # Multi-task Auxiliary Label Prediction
  for aux_task in spec.keys():
    aux_prediction = np.argmax(_logits[aux_task]) + 1
    print(spec[aux_task]['csv_index'] + ' Prediction: ' + str(aux_prediction))
