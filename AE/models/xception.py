import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Conv2D, Flatten


class FineTunedXception(tf.keras.Model):
    """BiT with a new head."""

    def __init__(self):
        super().__init__()
        self.to_rgb = Conv2D(filters=3, kernel_size=7, strides=(1, 1))
        self.xception = Xception(include_top=False, weights='imagenet')
        self.flatten_op = Flatten()
        self.head = tf.keras.layers.Dense(1, activation='sigmoid',
                                          kernel_initializer='zeros')  # zeros is very important

    def call(self, images, training=False):
        images_to_rgb = self.to_rgb(images, training=training)
        embeddings = self.xception(images_to_rgb, training=False)
        embeddings = self.flatten_op(embeddings)
        logits = self.head(embeddings, training=training)
        return logits


if __name__ == '__main__':
    PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
    if len(PHYSICAL_DEVICES) > 0:
        tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

    model = FineTunedXception()

    dummy_variable = tf.zeros((32, 128, 128, 2))
    logits = model(dummy_variable, training=False)
    print(model.summary())
