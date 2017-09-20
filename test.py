import tensorflow as tf
from nets import my_seg_net
import matplotlib.pyplot as plt

# #a=tf.Variable(tf.truncated_normal([None,1280,1918,3])) 
# a = tf.placeholder(dtype=tf.float32,shape=[4,1280,1920,3])
# logits,endpoints = my_seg_net.my_seg_net1(a,2)
# print(endpoints)

import pandas as pd
import glob

files = glob.glob('/home/xiaoyu/Documents/segmentation/datasets/data/submission/*.csv')
files.sort()
files = sorted(files, key=lambda d : int(d.split('_')[-1].split('.')[0]))
names=[]
rles=[]
for i in files:
    df_test = pd.read_csv(i)
    print(i)
    print('sample_submission.csv shape:: ', df_test.shape)
    print('sample_submission.csv columns:: ', df_test.columns.values.tolist())
    names.extend(df_test['img'])
    rles.extend(df_test['rle_mask'])



print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submission.csv.gz', index=False, compression='gzip')