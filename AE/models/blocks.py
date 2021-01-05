import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, ReLU, MaxPool2D, Activation, \
    SeparableConv2D, add
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2


class BasicEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, padding='same'):
        super(BasicEncoderBlock, self).__init__()
        self.block = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
            MaxPool2D()
        ])

    def call(self, inputs, training=False):
        return self.block(inputs, training=training)


class BasicDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, padding='same'):
        super(BasicDecoderBlock, self).__init__()
        self.block = Sequential([
            UpSampling2D(),
            Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
        ])

    def call(self, inputs, training=False):
        return self.block(inputs, training=training)


class XceptionEntryBlock(tf.keras.layers.Layer):
    def __init__(self, filters=32):
        super(XceptionEntryBlock, self).__init__()
        self.block = Sequential([
            Conv2D(filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block1_conv1'),
            BatchNormalization(name='block1_conv1_bn'),
            Activation('relu', name='block1_conv1_act'),
            Conv2D(filters, (3, 3), padding='same', use_bias=False, name='block1_conv2'),
            BatchNormalization(name='block1_conv2_bn'),
            Activation('relu', name='block1_conv2_act')
        ])

    def call(self, inputs, training=False):
        output = self.block(inputs, training=training)
        return output


class XceptionEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64, activity_regularizer=None):
        # for sparse autoencoding
        if activity_regularizer is not None:
            activity_regularizer = L1L2(l1=activity_regularizer)

        super(XceptionEncoderBlock, self).__init__()
        self.residual = Sequential([
            Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization()
        ])
        self.block = Sequential([
            SeparableConv2D(filters, (3, 3), padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,
                            activity_regularizer=activity_regularizer),
            BatchNormalization(),
            MaxPool2D((3, 3), strides=(2, 2), padding='same')
        ])

    def call(self, inputs, training=False):
        residual = self.residual(inputs, training=training)
        x = self.block(inputs, training=training)
        output = add([x, residual])
        return output


class XceptionDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64):
        super(XceptionDecoderBlock, self).__init__()
        self.residual = Sequential([
            UpSampling2D(),
            Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False),
            BatchNormalization()
        ])
        self.block = Sequential([
            UpSampling2D(),
            SeparableConv2D(filters, (3, 3), padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            SeparableConv2D(filters, (3, 3), padding='same', use_bias=False),
            BatchNormalization(),
        ])

    def call(self, inputs, training=False):
        residual = self.residual(inputs, training=training)
        x = self.block(inputs, training=training)
        output = add([x, residual])
        return output


class XceptionFinalBlock(tf.keras.layers.Layer):
    def __init__(self, filters=32):
        super(XceptionFinalBlock, self).__init__()
        self.block = Sequential([
            Conv2D(filters, (3, 3), use_bias=False, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            UpSampling2D(),
            Conv2D(filters, (3, 3), use_bias=False, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(1, (1, 1), use_bias=True, padding='same'),
        ])

    def call(self, inputs, training=False):
        return self.block(inputs, training=training)
