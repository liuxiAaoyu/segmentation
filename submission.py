## Modified Version from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523
import pandas as pd

import tensorflow as tf
import numpy as np
import glob
import cv2

slim = tf.contrib.slim

from nets import my_seg_net
from preprocessing import carvana_preprocessing

## Import required info from your training script
from skimage.transform import resize

## Used to save time
from multiprocessing import Pool

import time
import gc

## Configure with number of CPUs you have or the number of processes to spin ##
CPUs = 2
INPUT_SHAPE = (1918,1280)
## Tune it; used in generator
batch_size = 1

## Mask properties
WIDTH_ORIG = 1918
HEIGHT_ORIG = 1280

## More Tuning
MASK_THRESHOLD = 0.6

## Submission data
df_test = pd.read_csv('./datasets/data/sample_submission.csv')
print('sample_submission.csv shape:: ', df_test.shape)
print('sample_submission.csv columns:: ', df_test.columns.values.tolist())
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))

#tf.contrib.keras.

## https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
def run_length_encode(img):
    img = cv2.resize(img, (WIDTH_ORIG, HEIGHT_ORIG))
    flat_img = img.flatten()
    flat_img[0] = 0
    flat_img[-1] = 0
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding.strip()

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
#predictions, localisations, _, orientations, _ = ssd_net.net(
#        image_4d, is_training=False, dropout_keep_prob=1., update_feat_shapes=True, reuse=reuse)
_tv=slim.get_variables()
for i in _tv:
    print(i)
# Restore model.
#ckpt_filename = '/home/xiaoyu/logs/ssd_300_kitti./model.ckpt-226057'
#ckpt_filename = '/home/xiaoyu/Documents/segmentation/log/model.ckpt-15980'
ckpt_filename = '/home/xiaoyu/Documents/segmentation/log3/model.ckpt-58404'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
#ckpt_filename = '/home/xiaoyu/catkin_ws/src/mychallenge/models/lidarmodel.ckpt'
#saver.save(isess, ckpt_filename)


# Main image processing routine.
def process_image(sess, img):

    im_softmax = sess.run([tf.nn.softmax(logits)],{img_input: img})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > MASK_THRESHOLD).reshape(image_shape[0], image_shape[1], 1)

    #mask = np.dot(segmentation, np.array([[ 0, 255, 0]]))
    #mask = np.asarray(mask,dtype=np.uint8)
    # myimg = cv2.resize(img,(image_shape[1],image_shape[0]))

    # result = cv2.addWeighted(myimg, 1, mask, 1, 0)
    # out=np.dstack((result[:,:,2],result[:,:,1],result[:,:,0]))
    # cv2.imshow('image', out)#cv2.pyrDown(out))
    # cv2.waitKey(1)
    mask = im_softmax.reshape(image_shape[0], image_shape[1], 1)
    return mask


def inference():

    #cv2.namedWindow("image")
    #cv2.startWindowThread()
    #when first time allocate GPU memory it will take about 0.2 second 
    test = np.zeros((1280,1918,3),dtype=np.uint8)
    process_image(isess,test)
    files = glob.glob('/home/xiaoyu/Documents/segmentation/datasets/data/test/*.jpg')
    files.sort()

    re=5
    i=re*1000+1
    for j in range(re:100):
        filenames = names[1000*j:1000*(j+1)]
        rles = []
        for f in filenames:
            f = '/home/xiaoyu/Documents/segmentation/datasets/data/test/'+f
            img = cv2.imread(f)
            mask = process_image(isess,img)
            split_rle = run_length_encode(mask)
            rles.append(split_rle)
            print(i)
            i+=1

        print("Generating submission file...")
        df = pd.DataFrame({'img': filenames, 'rle_mask': rles})
        df.to_csv('./datasets/data/submission/submission_%d.csv'%j, index=False, compression=None)
    print('finished')


if __name__ == "__main__":
    inference()
