import tensorflow as tf

AUGMENTATIONS = {
    0: lambda x: x,
    1: lambda x: tf.image.random_flip_left_right(x),
    2: lambda x: tf.image.random_brightness(x, 0.2),
    3: lambda x: tf.image.random_contrast(x, 0, 2)
}



def augment(images):
    # loop over each image image in sequence so the augmentations will be diverse in batch

    # make a list of the tensors since TF can't handle item assignment (fail)
    tensors = tf.unstack(images, axis=0)
    for i in range(len(tensors)):
        # random invert 50/50
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5:
            tensors[i] = -tensors[i]

        augment_fn = AUGMENTATIONS[tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32).numpy()]
        tensors[i] = augment_fn(tensors[i])

    images = tf.stack(tensors, axis=0)
    return images


def random_gamma(x, max_gain=1, max_gamma=2):
    gamma = tf.random.uniform(shape=[], minval=0.0, maxval=max_gamma, dtype=tf.float32)
    gain = tf.random.uniform(shape=[], minval=0.0, maxval=max_gain, dtype=tf.float32)
    return tf.clip_by_value(tf.image.adjust_gamma(x, gamma=gamma, gain=gain), clip_value_min=0, clip_value_max=1)