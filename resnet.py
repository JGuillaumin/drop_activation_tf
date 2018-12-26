"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from drop_activation import DropActivationKeras as DropActivation
from randomized_relu import RandomizedReLUKeras as RandomizedReLU


class ResNet56:
    def __init__(self,
                 input_shape=None,
                 classes=10,
                 p=0.95,
                 rate=0.4,
                 activation='drop-activation',
                 a=3,
                 b=8,
                 **kwargs):

        list_activations = ['drop-activation', 'relu', 'randomized-relu', 'relu-dropout']
        if activation not in list_activations:
            raise ValueError("Invalid activation function : {} ! Must be in {}".format(activation, list_activations))

        self.input_shape = input_shape

        self.classes = classes
        self.activation = activation
        self.p = p
        self.rate = rate
        self.a = a
        self.b = b

        if backend.image_data_format() == 'channels_last':
            self.bn_axis = 3
        else:
            self.bn_axis = 1

        self.block_sizes = [9, 9, 9]
        self.block_strides = [1, 2, 2]
        self.init_num_filters = 16
        self.momentum_bn = 0.997
        self.epsilon_bn = 0.00001

        self.weight_decay = 0.0002

    def build_model(self):

        img_input = layers.Input(shape=self.input_shape)
        x = img_input

        x = layers.Conv2D(self.init_num_filters, (3, 3),
                          strides=(1, 1),
                          padding='valid',
                          use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.weight_decay))(x)

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.init_num_filters * (2**i)

            x = self.residual_block_v2(x, num_filters, with_projection=True, strides=self.block_strides[i])

            for _ in range(1, num_blocks):
                x = self.residual_block_v2(x, num_filters, with_projection=False, strides=1)

        x = layers.BatchNormalization(axis=self.bn_axis, momentum=self.momentum_bn, epsilon=self.epsilon_bn)(x)
        x = self.activation_block(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(self.classes, activation='softmax',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(x)

        # Create model.
        model = Model(img_input, x, name='resnet50')

        return model

    def activation_block(self, inputs):
        if self.activation == "relu":
            return layers.Activation('relu')(inputs)
        elif self.activation == "drop-activation":
            return DropActivation(p=self.p)(inputs)
        elif self.activation == "relu-dropout":
            x = layers.Dropout(rate=self.rate)(inputs)
            x = layers.Activation("relu")(x)
            return x
        elif self.activation == "randomized-relu":
            return RandomizedReLU(a=self.a, b=self.b)(inputs)
        else:
            # linear activation
            return inputs

    def residual_block_v2(self, inputs, num_filters, with_projection=False, strides=1):

        shortcut = inputs

        inputs = layers.BatchNormalization(axis=self.bn_axis, momentum=self.momentum_bn,
                                           epsilon=self.epsilon_bn)(inputs)
        inputs = self.activation_block(inputs)

        if with_projection:
            shortcut = layers.Conv2D(num_filters, (1, 1),
                                     strides=(strides, strides),
                                     padding='same' if strides == 1 else 'valid',
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=regularizers.l2(self.weight_decay),
                                     use_bias=False)(shortcut)

        if strides > 1:
            inputs = layers.ZeroPadding2D(padding=(1, 1))(inputs)

        inputs = layers.Conv2D(num_filters, (3, 3),
                               strides=(strides, strides),
                               padding='same' if strides == 1 else 'valid',
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(self.weight_decay),
                               use_bias=False)(inputs)
        inputs = layers.BatchNormalization(axis=self.bn_axis, momentum=self.momentum_bn,
                                           epsilon=self.epsilon_bn)(inputs)
        inputs = self.activation_block(inputs)

        inputs = layers.Conv2D(num_filters, (3, 3),
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(self.weight_decay),
                               use_bias=False)(inputs)

        inputs = layers.add([inputs, shortcut])
        return inputs
