import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import os
import random
from dataset_utils import int64_feature, float_feature, bytes_feature, string_feature, _EncodedFloatFeature,_EncodedBytesFeature
import matplotlib.pyplot as plt

# im = Image.open('/home/xiaoyu/Documents/segmentation/datasets/data/train_masks/00087a6bd4dc_01_mask.gif')
# im = np.asarray(im)
# #im = im[ np.newaxis]

# print(im)
# print(im.shape)
def im_to_encode(sess, fname):
    im = Image.open(fname)
    im = np.asarray(im)

    input_image = tf.placeholder(dtype = tf.uint8)
    input_image_extend = tf.expand_dims(input_image, -1)
    # #a = tf.convert_to_tensor(a)
    encode = tf.image.encode_png(input_image_extend)
    def ndarray2png_fn(sess, image):
        image_data = sess.run(encode,feed_dict={input_image : image})
        #image_data = sess.run(a,feed_dict={input_image : image})
        return image_data
    re = ndarray2png_fn(sess, im)
    return re
# with tf.Session() as sess:
#     image_png = ndarray2png_fn(sess, im)


mask_placeholder = tf.placeholder(dtype = tf.string)
gif_image = tf.image.decode_gif(mask_placeholder)
png_image = tf.squeeze(gif_image, [0])
png_image = png_image/255
_, _, mask_float = tf.split(png_image, [1,1,1], axis=2)
png_image = tf.cast(png_image,dtype = tf.uint8)
mask_raw = png_image#.tostring()
png_encode = tf.image.encode_png(png_image)
jpeg_encode = tf.image.encode_jpeg(png_image,quality=1)
def gif_to_png(sess, mask):
    png = sess.run(png_encode, feed_dict = {mask_placeholder:mask})
    # print(png.shape)
    # plt.imshow(png)
    # plt.show()
    return png
def gif_to_float(sess,mask):
    ffloat = sess.run(mask_float, feed_dict = {mask_placeholder:mask})
    return ffloat
def gif_to_raw(sess,mask):
    raw = sess.run(mask_raw, feed_dict = {mask_placeholder:mask})
    return raw.tostring()
def gif_to_jpeg(sess, mask):
    jpeg = sess.run(jpeg_encode, feed_dict = {mask_placeholder:mask})
    return jpeg

img_placeholder = tf.placeholder(dtype = tf.string)
image = tf.image.decode_jpeg(img_placeholder)
#_image = tf.cast(png_image,dtype = tf.uint8)
image_jpeg_encode = tf.image.encode_jpeg(image)
image_png_encode = tf.image.encode_png(image)
def jpeg_to_encode(sess, image):
    jpeg = sess.run(image_png_encode, feed_dict = {img_placeholder:image})
    # print(png.shape)
    # plt.imshow(png)
    # plt.show()
    return jpeg

mask_uint8 = tf.cast(mask_float, tf.uint8)
image_contact = tf.concat([image,mask_uint8], axis = 2)
contact_png_encode = tf.image.encode_png(image_contact)
def jpeg_gif_to_encode(sess,jpeg,gif):
    re = sess.run(contact_png_encode, feed_dict = {img_placeholder:jpeg,mask_placeholder:gif})
    return re



def decode_image_mask(image_mask_data, sess):
    pass
 
def process_data(image, image_mask, tfrecord_write):
    image_format = b'PNG'
    image_mask_format = b'PNG'
    xxx = random.random()
    example = tf.train.Example( features = tf.train.Features(
        feature ={
            'image/encoded' : bytes_feature(image),
            'image/format' : bytes_feature(image_format),
            #'mask/encode' :bytes_feature(image_mask),
            #'mask/format' :bytes_feature(image_mask_format),
            #'mask/encode' :_EncodedFloatFeature(image_mask),
        }
    ))
    tfrecord_write.write(example.SerializeToString())

def process_data1(image,  tfrecord_write):
    image_format = b'PNG'
    example = tf.train.Example( features = tf.train.Features(
        feature ={
            'image/encoded' : bytes_feature(image),
            'image/format' : bytes_feature(image_format),
        }
    ))
    tfrecord_write.write(example.SerializeToString())

# mask = tf.gfile.FastGFile('/home/xiaoyu/Documents/segmentation/datasets/data/train_masks/00087a6bd4dc_01_mask.gif','rb').read()
# with tf.Session() as sess:
#    (gif_to_png(sess,mask))
#    f=gif_to_float(sess,mask)
#    print(type(f))
#    print(f.shape)
# #print(mask)
# image = tf.gfile.FastGFile('/home/xiaoyu/Documents/segmentation/datasets/data/train/00087a6bd4dc_01.jpg','rb').read()
# #print(image)


files = glob.glob('/home/xiaoyu/Documents/segmentation/datasets/data/train/*.jpg')
files.sort()
train = files[:4592]
valid = files[4592:]
random.shuffle(train)

i=0
with tf.Session() as sess:
    with tf.python_io.TFRecordWriter('./datasets/data/carvana_train4592.tfrecord') as tfrecord_writer:
      for f in train:
        fname = os.path.split(f)[1]
        gtname = '/home/xiaoyu/Documents/segmentation/datasets/data/train_masks/' + os.path.splitext(fname)[0] + '_mask.gif'
        mask = tf.gfile.FastGFile(gtname,'rb').read()
        image = tf.gfile.FastGFile(f,'rb').read()
        cimage = jpeg_gif_to_encode(sess, image, mask)
        process_data1(cimage, tfrecord_writer)
        i+=1
        print(i)
    print('train finished')

    with tf.python_io.TFRecordWriter('./datasets/data/carvana_valid496.tfrecord') as tfrecord_writer:#./datasets/data/carvana_valid496.tfrecord
      for f in valid:
        fname = os.path.split(f)[1]
        gtname = '/home/xiaoyu/Documents/segmentation/datasets/data/train_masks/' + os.path.splitext(fname)[0] + '_mask.gif'
        mask = tf.gfile.FastGFile(gtname,'rb').read()
        image = tf.gfile.FastGFile(f,'rb').read()
        cimage = jpeg_gif_to_encode(sess, image, mask)
        process_data1(cimage, tfrecord_writer)
        i+=1
        print(i)
    print('valid finished')

    with tf.python_io.TFRecordWriter('./datasets/data/train5088.tfrecord') as tfrecord_writer:
      random.shuffle(files)
      for f in files:
        mask = tf.gfile.FastGFile(gtname,'rb').read()
        image = tf.gfile.FastGFile(f,'rb').read()
        cimage = jpeg_gif_to_encode(sess, image, mask)
        process_data1(cimage, tfrecord_writer)
        i+=1
        print(i)
    print('file finished')