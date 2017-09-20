import tensorflow as tf
import numpy as np
import glob
import cv2

slim = tf.contrib.slim

from nets import my_seg_net
from preprocessing import carvana_preprocessing


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)


# Input placeholder.

image_shape = (640, 960)
#net_shape = (300, 749)

image_file = tf.placeholder(tf.string)
img_input = tf.placeholder(tf.uint8, shape=(1280, 1918, 3))
#img_input = tf.image.decode_jpeg(image_file,3)
image_pre = carvana_preprocessing.preprocess_for_eval(img_input, image_shape[0], image_shape[1])
image_4d = tf.expand_dims(image_pre, 0)

with slim.arg_scope(my_seg_net.my_arg_scpoe1()):
    logits, _ = my_seg_net.my_seg_net3(image_4d,2)

im_softmax = tf.nn.softmax(logits)
im_softmax = tf.reshape(im_softmax,shape=(image_shape[0], image_shape[1], 2))
_, im_softmax = tf.split(im_softmax, [1, 1], axis=2)

_tv=slim.get_variables()
for i in _tv:
    print(i)
# Restore model.
#ckpt_filename = '/home/xiaoyu/logs/ssd_300_kitti./model.ckpt-226057'
ckpt_filename = '/home/xiaoyu/Documents/segmentation/log3/model.ckpt-58404'
#ckpt_filename = '/home/xiaoyu/Documents/segmentation/log3_/model.ckpt-55328'#model.ckpt-58404'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
#ckpt_filename = '/home/xiaoyu/catkin_ws/src/mychallenge/models/lidarmodel.ckpt'
#saver.save(isess, ckpt_filename)

# inputs = tf.contrib.keras.layers.Input(tensor=img_input)
# pres = tf.contrib.keras.layers.Input(tensor=image_4d)
# outputs = tf.contrib.keras.layers.Input(tensor=im_softmax)
# model = tf.contrib.keras.models.Sequential()
# #model.inputs = inputs
# model.graph = isess.graph
# model.save('./my_model.h5') 

fnum = []
fnum.append(0)
pnum = []
pnum.append(0)
# Main image processing routine.
def process_image(sess, img):

    im = sess.run([im_softmax],{img_input: img})
    #im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im[0] > 0.5).reshape(image_shape[0], image_shape[1], 1)
    # mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    # mask = scipy.misc.toimage(mask, mode="RGBA")
    # street_im = scipy.misc.toimage(img)
    # street_im.paste(mask, box=None, mask=mask)
    mask = np.dot(segmentation, np.array([[ 0, 255, 0]]))
    mask = np.asarray(mask,dtype=np.uint8)
    myimg = cv2.resize(img,(image_shape[1],image_shape[0]))
    #result = 
    result = cv2.addWeighted(myimg, 1, mask, 1, 0)
    out=np.dstack((result[:,:,2],result[:,:,1],result[:,:,0]))
    cv2.imshow('image', out)#cv2.pyrDown(out))
    cv2.waitKey(0)

    return 


def inference():
    #cv2.namedWindow("image_heightmap")
    cv2.namedWindow("image")
    cv2.startWindowThread()
    #when first time allocate GPU memory it will take about 0.2 second 
    test = np.zeros((1280,1918,3),dtype=np.uint8)
    process_image(isess,test)
    files = glob.glob('/home/xiaoyu/Documents/segmentation/datasets/data/test/*.jpg')
    files.sort()
    for f in files:
        #f_img=tf.gfile.FastGFile(f,'rb').read()
        img = cv2.imread(f)
        process_image(isess,img)
    

if __name__ == "__main__":
    inference()
