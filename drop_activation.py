from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import layers as keras_layers

from tensorflow.python.layers import base

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn


class DropActivationKeras(keras_layers.Layer):
    def __init__(self, p=0.95, seed=None, **kwargs):
        super(DropActivationKeras, self).__init__(**kwargs)
        self.p = p
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs_training():

            with ops.name_scope("drop_activation_training"):
                x = ops.convert_to_tensor(inputs, name="x")
                if not x.dtype.is_floating:
                    raise ValueError("x has to be a floating point tensor since it's going to"
                                     " be scaled. Got a %s tensor instead." % x.dtype)
                if isinstance(self.p, numbers.Real) and not 0 < self.p <= 1:
                    raise ValueError("p must be a scalar tensor or a float in the "
                                     "range (0, 1], got %g" % self.p)

                # Early return is nothin to be dropped
                if isinstance(self.p, float) and self.p == 1.:
                    return nn.relu(x)
                if context.executing_eagerly():
                    if isinstance(self.p, ops.EagerTensor):
                        if self.p.numpy() == 1:
                            return nn.relu(x)

                else:
                    p = ops.convert_to_tensor(self.p, dtype=x.dtype, name="p")
                    p.get_shape().assert_is_compatible_with(tensor_shape.scalar())

                    # Do nothing if we know keep_prob == 1
                    if tensor_util.constant_value(p) == 1:
                        return nn.relu(x)

                    noise_shape = array_ops.shape(x)
                    random_tensor = 1 - p
                    random_tensor += random_ops.random_uniform(noise_shape, seed=self.seed, dtype=x.dtype)
                    # random_tensor ~ uniform distrib [1 - p, 2 - p), ex: [0.05, 1.05)

                    binary_tensor = math_ops.floor(random_tensor)
                    # in binary tensor ~ 5% of are set 1 , 95% are set 0

                    # drop 95% of the negative part, keep all in the positive part
                    # old implementation:
                    # ret = - binary_tensor*nn.relu((-x)) + nn.relu(x)
                    # new implementation, only 1 relu operation
                    ret = binary_tensor*x + (1-binary_tensor)*nn.relu(x)
                    if not context.executing_eagerly():
                        ret.set_shape(x.get_shape())
                    return ret

        def dropped_inputs_testing():
            with ops.name_scope("drop_activation_testing"):
                # in testing phase, deterministic activation function
                # leaky ReLU with slope = 1. - p
                return nn.leaky_relu(inputs, alpha=1-self.p)

        with ops.name_scope(self.name, "drop_activation", [inputs]):
            output = tf_utils.smart_cond(training,
                                         true_fn=dropped_inputs_training,
                                         false_fn=dropped_inputs_testing)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'p': self.p,
            'seed': self.seed
        }

        base_config = super(DropActivationKeras, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DropActivationTensorFlow(DropActivationKeras, base.Layer):
    def __init__(self, p=0.95, seed=None, name=None, **kwargs):
        super(DropActivationTensorFlow, self).__init__(p=p, seed=seed, name=name)

    def call(self, inputs, training=False):
        return super(DropActivationTensorFlow, self).call(inputs, training=training)


def drop_activation(inputs, p=0.95, training=False, seed=None, name=None):

    layer = DropActivationTensorFlow(p=p, seed=seed, name=name)
    return layer.call(inputs, training=training)