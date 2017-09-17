import tensorflow as tf
from nets import my_seg_net
import matplotlib.pyplot as plt
slim = tf.contrib.slim
#a=tf.Variable(tf.truncated_normal([None,1280,1918,3])) 
net = my_seg_net.my_seg_net4
a = tf.placeholder(dtype=tf.float32,shape=[4,net.default_image_height,net.default_image_width,3])
with slim.arg_scope(my_seg_net.my_arg_scpoe1()):
    logits,endpoints = my_seg_net.my_seg_net4(a,2)
for i in endpoints:
    print(endpoints[i])

