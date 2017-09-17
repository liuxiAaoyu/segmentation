import tensorflow as tf
from nets import inception_v4
from nets import inception_utils
import numpy as np

slim = tf.contrib.slim

def my_arg_scpoe1(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc
    
def load_inception_seg(sess, inception_path):
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.TRAINING],'/home/xiaoyu/Documents/tfmodels/inceptionv4')
    inputs = sess.graph.get_tensor_by_name('input:0')
    l4o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_4a/concat:0')
    l5o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_5e/concat:0')
    l6o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_6h/concat:0')
    l7o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_7d/concat:0')
    return inputs, l4o, l5o, l6o, l7o

def layers_inception(vgg_layer4_out, vgg_layer5_out, vgg_layer6_out, vgg_layer7_out, num_classes, is_training = False):

    deconv7 = tf.layers.conv2d_transpose(vgg_layer7_out,1024,4,2,'VALID')
    deconv7 = tf.layers.batch_normalization(deconv7)

    conv6 = tf.layers.conv2d(vgg_layer6_out,1,1, trainable=False)
    add1 = tf.add(deconv7,vgg_layer6_out)
    add1 = tf.layers.batch_normalization(add1)
    deconv6 = tf.layers.conv2d_transpose(add1,384,3,2,'VALID')
    deconv6 = tf.layers.batch_normalization(deconv6)

    conv5 = tf.layers.conv2d(vgg_layer5_out,1,1, trainable=False)
    add2 = tf.add(deconv6,vgg_layer5_out)
    add2 = tf.layers.batch_normalization(add2)
    deconv5 = tf.layers.conv2d_transpose(add2,192,4,2,'VALID')
    deconv5 = tf.layers.batch_normalization(deconv5)

    conv4 = tf.layers.conv2d(vgg_layer4_out,1,1, trainable=False)
    add3 = tf.add(deconv5,vgg_layer4_out)
    add3 = tf.layers.batch_normalization(add3)
    deconv4 = tf.layers.conv2d_transpose(add3,64,20,4,'VALID')
    deconv4 = tf.layers.conv2d_transpose(deconv4,num_classes,1,1,'SAME')

    return deconv4

def fcn_inception(inputs, num_classes, is_training = None, scope = None):
    net, end_points = inception_v4.inception_v4_base(inputs)

    layer1_out = end_points['Conv2d_1a_3x3']
    layer2_out = end_points['Conv2d_2b_3x3']
    layer3_out = end_points['Mixed_3a']
    layer4_out = end_points['Mixed_4a']
    layer5_out = end_points['Mixed_5e']
    layer6_out = end_points['Mixed_6h']
    layer7_out = end_points['Mixed_7d']

    with tf.variable_scope(scope, 'fcn_inception'):
        deconv7 = slim.conv2d_transpose(layer7_out,1024,4,2,'VALID',scope='deconv7')
        end_points["fcn_inception/deconv7"] = deconv7

        add6 = tf.add(deconv7,layer6_out,name='add6')
        deconv6 = slim.conv2d_transpose(add6,384,3,2,'VALID',scope='deconv6')
        end_points["deconv6"] = deconv6


        add5 = tf.add(deconv6,layer5_out,name='add5')
        deconv5 = slim.conv2d_transpose(add5,192,4,2,'VALID',scope='deconv5')
        end_points["deconv5"] = deconv5


        add4 = tf.add(deconv5,layer4_out,name='add4')
        deconv4 = slim.conv2d_transpose(add4,160,3,1,'VALID',scope='deconv4')
        end_points["deconv4"] = deconv4

        add3 = tf.add(deconv4,layer3_out,name='add3')
        deconv3 = slim.conv2d_transpose(add3,64,3,2,'VALID',scope='deconv3')
        end_points["deconv3"] = deconv3

        add2 = tf.add(deconv3,layer2_out,name='add2')
        deconv2 = slim.conv2d_transpose(add2,32,3,1,'VALID',scope='deconv2')
        end_points["deconv2"] = deconv2

        add1 = tf.add(deconv2,layer1_out,name='add1')
        deconv1 = slim.conv2d_transpose(add1,num_classes,4,2,'VALID',scope='deconv1')
        end_points["deconv1"] = deconv1

        logits = tf.reshape(deconv1,[-1, num_classes],name = 'logits')
    return logits, end_points

fcn_inception.default_image_size = 448
fcn_inception.default_image_height = 448#640#448#1280
fcn_inception.default_image_width = 448#960#448#1920
fcn_inception.arg_scope = my_arg_scpoe1

def unpool1(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape =  tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        #ret = tf.reshape(ret, tf.stack(output_shape))
        ret = tf.reshape(ret, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret


def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def my_seg_net1(inputs, num_classes = None, is_training = None, scope = None):
    end_points={}
    #last_layer, _ = load_inception(input_image, num_classes)
    with tf.variable_scope(scope, 'my_seg_net1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.conv2d_transpose],
                        stride=1, padding='SAME'):
            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1_1')
            end_points['block1_1'] = net
            net, ind1_1 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1_1')
            
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            end_points['block1_2'] = net
            net, ind1_2 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1_2')
            # Block 2.
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv2_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            end_points['block2'] = net
            net, ind2 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')
            # Block 3.
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_2_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            #net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            #net = slim.conv2d(net, 256, [1, 1], scope='conv3_3_1x1')
            end_points['block3'] = net
            net, ind3 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            end_points['block4'] = net

            net, ind4 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_2_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_3')
            end_points['block5'] = net

            net, ind5 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

            net = slim.conv2d_transpose(net,1024,[3,3],[2,2])
            #net = unpool(net,ind5,scope='unpool5')
            #net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_2_1x1')
            #net = slim.conv2d(net, 512, [3, 3], scope='dconv5_3')
            end_points['block5d'] = net

            net = slim.conv2d_transpose(net,512,[3,3],[2,2])
            #net = unpool(net,ind4,scope='unpool4')
            #net = slim.conv2d(net, 512, [3, 3], scope='dconv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_3')
            end_points['block4d'] = net

            net = slim.conv2d_transpose(net,256,[3,3],[2,2])
            #net = unpool(net,ind3,scope='unpool3')
            #net = slim.conv2d(net, 256, [3, 3], scope='dconv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_2_1x1')
            #net = slim.conv2d(net, 256, [3, 3], scope='dconv3_3')
            end_points['block3d'] = net

            net = slim.conv2d_transpose(net,128,[3,3],[2,2])
            #net = unpool(net,ind2,scope='unpool2')
            #net = slim.conv2d(net, 128, [3, 3], scope='dconv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='dconv2_1_1x1')
            #net = slim.conv2d(net, 128, [3, 3], scope='dconv2_2')
            end_points['block2d'] = net

            net = slim.conv2d_transpose(net,64,[3,3],[2,2])
            #net = unpool(net,ind1_2,scope='unpool1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='dconv1_2')
            end_points['block1_2d'] = net

            net = slim.conv2d_transpose(net,32,[3,3],[2,2])
            #net = unpool(net,ind1_1,scope='unpool1_1')
            net = slim.conv2d(net, num_classes, [3, 3], scope='dconv1_1')
            end_points['block1_1d'] = net

            last_layer = net#slim.conv2d(net, num_classes, [1, 1], scope='last')
            end_points['blocklast'] = last_layer

            logits = tf.reshape(last_layer,[-1, num_classes],name = 'logits')
            return logits, end_points

my_seg_net1.default_image_size = 448
my_seg_net1.default_image_height = 1280
my_seg_net1.default_image_width = 1920
my_seg_net1.my_seg_net1_arg_scope = my_arg_scpoe1


def my_seg_net2(inputs, num_classes = None, is_training = None, scope = None):
    end_points={}
    #last_layer, _ = load_inception(input_image, num_classes)
    with tf.variable_scope(scope, 'my_seg_net1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1_1')
            end_points['block1_1'] = net
            net, ind1_1 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1_1')
            
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            end_points['block1_2'] = net
            net, ind1_2 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1_2')
            # Block 2.
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv2_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            end_points['block2'] = net
            net, ind2 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')
            # Block 3.
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_2_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            #net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            #net = slim.conv2d(net, 256, [1, 1], scope='conv3_3_1x1')
            end_points['block3'] = net
            net, ind3 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            end_points['block4'] = net

            net, ind4 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_2_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_3')
            end_points['block5'] = net

            net, ind5 = tf.nn.max_pool_with_argmax(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

            #net = tf.layers.conv2d_transpose(net,1024,3,2,'SAME')
            net = unpool(net,ind5,scope='unpool5')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_2_1x1')
            #net = slim.conv2d(net, 512, [3, 3], scope='dconv5_3')
            end_points['block5d'] = net

            #net = tf.layers.conv2d_transpose(net,512,3,2,'SAME')
            net = unpool(net,ind4,scope='unpool4')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_2_1x1')
            #net = slim.conv2d(net, 512, [3, 3], scope='dconv4_3')
            end_points['block4d'] = net

            #net = tf.layers.conv2d_transpose(net,256,3,2,'SAME')
            net = unpool(net,ind3,scope='unpool3')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_2_1x1')
            #net = slim.conv2d(net, 256, [3, 3], scope='dconv3_3')
            end_points['block3d'] = net

            #net = tf.layers.conv2d_transpose(net,128,3,2,'SAME')
            net = unpool(net,ind2,scope='unpool2')
            net = slim.conv2d(net, 128, [3, 3], scope='dconv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='dconv2_1_1x1')
            #net = slim.conv2d(net, 128, [3, 3], scope='dconv2_2')
            end_points['block2d'] = net

            #net = tf.layers.conv2d_transpose(net,64,3,2,'SAME')
            net = unpool(net,ind1_2,scope='unpool1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='dconv1_2')
            end_points['block1_2d'] = net

            #net = tf.layers.conv2d_transpose(net,32,3,2,'SAME')
            net = unpool(net,ind1_1,scope='unpool1_1')
            net = slim.conv2d(net, num_classes, [3, 3], scope='dconv1_1')
            end_points['block1_1d'] = net

            last_layer = net#slim.conv2d(net, num_classes, [1, 1], scope='last')
            end_points['blocklast'] = last_layer

            logits = tf.reshape(last_layer,[-1, num_classes],name = 'logits')
            return logits, end_points

my_seg_net2.default_image_size = 448
my_seg_net2.default_image_height = 1280
my_seg_net2.default_image_width = 1920
my_seg_net2.my_seg_net1_arg_scope = my_arg_scpoe1



def my_seg_net3(inputs, num_classes = None, is_training = None, scope = None):
    end_points={}
    #last_layer, _ = load_inception(input_image, num_classes)
    with tf.variable_scope(scope, 'my_seg_net3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        stride=1, padding='SAME'):

            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1_1')
            end_points['block1_1'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_1')
            
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            end_points['block1_2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_2')
            # Block 2.
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv2_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            # Block 3.
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='conv3_2_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='conv4_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            end_points['block4'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='conv5_2_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv5_3')
            end_points['block5'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool5')
            #Block 6
            net = slim.conv2d(net, 2048, [3, 3], scope='conv6_1')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv6_1_1x1')
            net = slim.conv2d(net, 2048, [3, 3], scope='conv6_2')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv6_2_1x1')
            net = slim.conv2d(net, 2048, [3, 3], scope='conv6_3')
            end_points['block6'] = net

            net = slim.conv2d_transpose(net,1024,[3,3],[2,2], scope='up5')
            net = tf.concat([net, end_points['block5']], axis=3, name='concat5')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_1')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_2')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv5_2_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv5_3')
            end_points['block5d'] = net

            net = slim.conv2d_transpose(net,512,[3,3],[2,2], scope='up4')
            net = tf.concat([net, end_points['block4']], axis=3, name='concat4')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_1')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_2')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv4_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv4_3')
            end_points['block4d'] = net

            net = slim.conv2d_transpose(net,256,[3,3],[2,2], scope='up3')
            net = tf.concat([net, end_points['block3']], axis=3, name='concat3')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv3_1')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv3_2')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv3_2_1x1')
            #net = slim.conv2d(net, 256, [3, 3], scope='dconv3_3')
            end_points['block3d'] = net

            net = slim.conv2d_transpose(net,256,[3,3],[2,2], scope='up2')
            net = tf.concat([net, end_points['block2']], axis=3, name='concat2')
            net = slim.conv2d(net, 128, [3, 3], scope='dconv2_1')
            net = slim.conv2d(net, 64, [1, 1], scope='dconv2_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='dconv2_2')
            end_points['block2d'] = net

            net = slim.conv2d_transpose(net,64,[3,3],[2,2], scope='up1_2')
            net = tf.concat([net, end_points['block1_2']], axis=3, name='concat1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='dconv1_2')
            end_points['block1_2d'] = net

            net = slim.conv2d_transpose(net,32,[3,3],[2,2], scope='up1_1')
            net = tf.concat([net, end_points['block1_1']], axis=3, name='concat1_1')
            net = slim.conv2d(net, num_classes, [3, 3], scope='dconv1_1')
            end_points['block1_1d'] = net

            last_layer = net#slim.conv2d(net, num_classes, [1, 1], scope='last')
            end_points['blocklast'] = last_layer

            logits = tf.reshape(last_layer,[-1, num_classes],name = 'logits')
            return logits, end_points

my_seg_net3.default_image_size = 448
my_seg_net3.default_image_height = 640#1280/2
my_seg_net3.default_image_width = 960#1920/2
my_seg_net3.my_seg_net1_arg_scope = my_arg_scpoe1




def my_seg_net4(inputs, num_classes = None, is_training = None, scope = None):
    end_points={}
    #last_layer, _ = load_inception(input_image, num_classes)
    with tf.variable_scope(scope, 'my_seg_net3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
            net = inputs
            # net = slim.conv2d(net, 16, [3, 3], scope='conv1_1_1')
            # net = slim.conv2d(net, 16, [3, 3], scope='conv1_1_2')
            # end_points['block1_1'] = net
            # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_1')
            
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2_2')
            end_points['block1_2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_2')
            # Block 2.
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [1, 1], scope='conv2_1_1x1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            # Block 3.
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv3_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv4_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_2')
            end_points['block4'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
            # Block 5.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv5_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            end_points['block5'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool5')
            #Block 6
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv6_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_2')
            end_points['block6'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool6')
            #Block 7
            net = slim.conv2d(net, 2048, [3, 3], scope='conv7_1')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7_1_1x1')
            net = slim.conv2d(net, 2048, [3, 3], scope='conv7_2')
            end_points['block7'] = net

            net = slim.conv2d_transpose(net,1024,[3,3],[2,2], scope='up6')
            net = tf.concat([net, end_points['block6']], axis=3, name='concat6')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv6_1')
            net = slim.conv2d(net, 512, [1, 1], scope='dconv6_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='dconv6_2')
            end_points['block6d'] = net

            net = slim.conv2d_transpose(net,512,[3,3],[2,2], scope='up5')
            net = tf.concat([net, end_points['block5']], axis=3, name='concat5')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv5_1')
            net = slim.conv2d(net, 256, [1, 1], scope='dconv5_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='dconv5_2')
            end_points['block5d'] = net

            net = slim.conv2d_transpose(net,256,[3,3],[2,2], scope='up4')
            net = tf.concat([net, end_points['block4']], axis=3, name='concat4')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv4_1')
            net = slim.conv2d(net, 128, [1, 1], scope='dconv4_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='dconv4_2')
            end_points['block4d'] = net

            net = slim.conv2d_transpose(net,128,[3,3],[2,2], scope='up3')
            net = tf.concat([net, end_points['block3']], axis=3, name='concat3')
            net = slim.conv2d(net, 128, [3, 3], scope='dconv3_1')
            net = slim.conv2d(net, 64, [1, 1], scope='dconv3_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='dconv3_2')
            end_points['block3d'] = net

            net = slim.conv2d_transpose(net,64,[3,3],[2,2], scope='up2')
            net = tf.concat([net, end_points['block2']], axis=3, name='concat2')
            net = slim.conv2d(net, 64, [3, 3], scope='dconv2_1')
            net = slim.conv2d(net, 32, [1, 1], scope='dconv2_1_1x1')
            net = slim.conv2d(net, 64, [3, 3], scope='dconv2_2')
            end_points['block2d'] = net

            net = slim.conv2d_transpose(net,32,[3,3],[2,2], scope='up1_2')
            net = tf.concat([net, end_points['block1_2']], axis=3, name='concat1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='dconv1_2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='dconv1_2_2')
            end_points['block1_2d'] = net

            # net = slim.conv2d_transpose(net,16,[3,3],[2,2], scope='up1_1')
            # net = tf.concat([net, end_points['block1_1']], axis=3, name='concat1_1')
            # net = slim.conv2d(net, 16, [3, 3], scope='dconv1_1_2')
            # net = slim.conv2d(net, num_classes, [3, 3], scope='dconv1_1_1')
            # end_points['block1_1d'] = net

            last_layer = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='last')
            end_points['blocklast'] = last_layer

            logits = tf.reshape(last_layer,[-1, num_classes],name = 'logits')
            return logits, end_points

my_seg_net4.default_image_size = 448
my_seg_net4.default_image_height = 640#1280/2
my_seg_net4.default_image_width = 960#1920/2
my_seg_net4.my_seg_net1_arg_scope = my_arg_scpoe1