import tensorflow as tf
from nets import my_seg_net
import matplotlib.pyplot as plt

#a=tf.Variable(tf.truncated_normal([None,1280,1918,3])) 
a = tf.placeholder(dtype=tf.float32,shape=[4,1280,1920,3])
logits,endpoints = my_seg_net.my_seg_net1(a,2)
print(endpoints)

