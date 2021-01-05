import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

class BiT(tf.keras.Model):
    """BiT with a new head."""

    def __init__(self, model_type='r50', train_bit=False):
        super().__init__()

        # handle relative path
        base_path = Path(__file__).parent
        path_to_bit_model = f'../checkpoints_skeleton/models/bit_m-{model_type}_1'
        resolved_path = (base_path / path_to_bit_model).resolve()

        # load model and attach head
        self.bit = hub.KerasLayer(str(resolved_path), trainable=train_bit)
        self.head = tf.keras.layers.Dense(1, activation='sigmoid',
                                          kernel_initializer='zeros')  # zeros is very important

    def call(self, images, training=False):
        embeddings = self.bit(images, training=training)
        logits = self.head(embeddings, training=training)
        return logits


if __name__ == '__main__':
    PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
    if len(PHYSICAL_DEVICES) > 0:
        tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

    model = BiT('r101x1', train_bit=False)

    dummy_variable = tf.zeros((2, 128, 128, 3))
    logits = model(dummy_variable, training=False)
    print(logits)
    print(model.summary())
