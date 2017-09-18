import tensorflow as tf
from nets import my_seg_net
import matplotlib.pyplot as plt

#a=tf.Variable(tf.truncated_normal([None,1280,1918,3])) 
# a = tf.placeholder(dtype=tf.float32,shape=[4,1280,1920,3])
# logits,endpoints = my_seg_net.my_seg_net1(a,2)
# print(endpoints)

from datasets import dataset_factory

slim = tf.contrib.slim
reader = tf.TFRecordReader()  
filename_queue = tf.train.string_input_producer(['/home/xiaoyu/Documents/segmentation/datasets/data/carvana_valid496.tfrecord'])  
_, serialized_example = reader.read(filename_queue)  
features = tf.parse_single_example(serialized_example, features={  
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      #'mask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      #'mask/format': tf.FixedLenFeature((), tf.string, default_value=''),
      #'xx': tf.FixedLenFeature([1], tf.float32),
    })  
decoder = slim.tfexample_decoder.Image( 'image/encoded', 'image/format',channels=4)
image = decoder.tensors_to_item(features)
# mask = tf.image.decode_png(features['mask/encoded'])
# mask_decoder = slim.tfexample_decoder.Image( 'mask/encoded', 'mask/format')
# maskd = mask_decoder.tensors_to_item(features)
# formats = features['mask/format']
# decoder =  slim.tfexample_decoder.Tensor('xx')
# xx = decoder.tensors_to_item(features)

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(100):
        a = sess.run(image)
        print(a.shape)
        #print(a)
        plt.imshow(a[:,:,:3])
        plt.show()  
        plt.imshow(a[:,:,3])
        plt.show()  
    coord.request_stop()
    coord.join(threads)   