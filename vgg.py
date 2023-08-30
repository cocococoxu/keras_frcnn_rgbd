# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers import Concatenate, Add, Reshape, Multiply
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
import keras.layers as KL
# CABM
channel_axis = 1 if K.image_data_format() == "channels_first" else 3
# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])


def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)    

def nn_base(input_tensor_one=None, input_tensor_two=None, trainable=False):


    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor_one is None:
        img_input_one = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor_one):
            img_input_one = Input(tensor=input_tensor_one, shape=input_shape)
        else:
            img_input_one = input_tensor_one

    if input_tensor_two is None:
        img_input_two = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor_two):
            img_input_two = Input(tensor=input_tensor_two, shape=input_shape)
        else:
            img_input_two = input_tensor_two

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_one')(img_input_one)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_one')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_one')(x1)

    # Block 2
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_one')(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_one')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_one')(x1)

    # Block 3
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_one')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_one')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_one')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_one')(x1)

    # Block 4
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_one')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2_one')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3_one')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_one')(x1)

    # Block 5
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_one')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_one')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_one')(x1)
    # x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x1)

    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_two')(img_input_two)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_two')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_two')(x2)

    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_two')(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_two')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_two')(x2)

    # Block 3
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_two')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_two')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_two')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_two')(x2)

    # Block 4
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_two')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2_two')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3_two')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_two')(x2)

    # Block 5
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_two')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_two')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_two')(x2)
    # x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x2)

    x = Concatenate()([x1, x2])
    # x = Add()([x1, x2])
    x = cbam_module(x)


    return x

def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois,7,7,512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,512,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


