import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

from models.blocks import BasicEncoderBlock, BasicDecoderBlock
from models.blocks import XceptionEntryBlock, XceptionDecoderBlock, XceptionEncoderBlock, XceptionFinalBlock


class Autoencoder(tf.keras.models.Model):
    """
    tensowflow.keras.models.Model class to be used as a convenient way to initialize all the moving parts and submodels
    necessary for running an Autoencoder architecture
    The model architecture is adaptive to the input_shape but (128, 128, c), (64, 64, c) or (32, 32, c) are recommended
    for optimal results.
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder(x) model
        self.Ex = self.make_encoder(name='Ex')

        # Generator(z) / Decoder(z) model
        self.Gz = self.make_decoder(name='Gz')

    def call(self, inputs, training=None):
        z = self.Ex(inputs, training=training)
        x_hat = self.Gz(z, training=training)
        return x_hat

    def make_encoder(self, name='Ex'):
        model = Sequential(
            [
                BasicEncoderBlock(32),
                BasicEncoderBlock(64),
                BasicEncoderBlock(64),
            ],
            name=name
        )
        return model

    def make_decoder(self, name='Gz'):
        model = tf.keras.Sequential(
            [
                BasicDecoderBlock(64),
                BasicDecoderBlock(64),
                BasicDecoderBlock(32),
                Conv2D(filters=1, kernel_size=1, padding='same', activation='tanh'),
            ],
            name=name
        )
        return model


class XAutoencoder(tf.keras.models.Model):
    """
    tensowflow.keras.models.Model class to be used as a convenient way to initialize all the moving parts and submodels
    necessary for running an Autoencoder architecture
    The model architecture is adaptive to the input_shape but (128, 128, c), (64, 64, c) or (32, 32, c) are recommended
    for optimal results.
    """

    def __init__(self, activity_regularizer=None):
        super(XAutoencoder, self).__init__()

        # Encoder(x) model Sequential

        self.Ex = self.make_encoder(activity_regularizer, name='Ex')

        # Generator(z) / Decoder(z) model Sequential
        self.Gz = self.make_decoder(name='Gz')

    def call(self, inputs, training=False):
        z = self.Ex(inputs, training=training)
        x_hat = self.Gz(z, training=training)
        x_hat = tf.nn.tanh(x_hat)
        return x_hat

    def make_encoder(self, activity_regularizer=None, name='Ex'):
        model = Sequential(
            [
                XceptionEntryBlock(32),
                XceptionEncoderBlock(64),
                XceptionEncoderBlock(128),
                XceptionEncoderBlock(256, activity_regularizer),
            ],
            name=name
        )
        return model

    def make_decoder(self, name='Gz'):
        model = tf.keras.Sequential(
            [
                XceptionDecoderBlock(256),
                XceptionDecoderBlock(128),
                XceptionDecoderBlock(64),
                XceptionFinalBlock(32),
            ],
            name=name
        )
        return model


if __name__ == '__main__':
    PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
    if len(PHYSICAL_DEVICES) > 0:
        tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

    xae = XAutoencoder()
    dummy_variable = tf.zeros((1, 256, 256, 1))
    x_hat = xae(dummy_variable, training=False)
    print(xae.summary())
