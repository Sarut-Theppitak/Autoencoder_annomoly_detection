import glob
import random

import numpy as np
import yaml
from tensorflow.keras.utils import to_categorical

from dataloader.annotation_utils import separate_images_by_label
from dataloader.processing.slicing import display_slices, slice_and_label
# from annotation_utils import separate_images_by_label
# from processing.slicing import display_slices, slice_and_label


def crop_generator(img_ann_list, batch_size=64, crop_size=256, roi=None,
                   shuffle=True, normalize=True, resize=1.0, repeat=True):
    """
    Given a list of tuples (img_path, ann_path) yield batches of img_batch, label_batches.
    Can shuffle the data, normalize the images, resize and be one-shot or infinite.
    :param list img_ann_list: list of tuples (str img_path, str ann_path)
    :param int batch_size: max batch size to yield (max may not be reached if bigger than img slices by crops
    :param int crop_size: in what crop_size x crops_size shapes to slice the image (assuming it's bigger)
    :param bool shuffle: whether to shuffle the data
    :param bool normalize: whether to transform [0, 255] images to [-1, 1]
    :param bool or float resize: resize the slices after cropping (usually to shrink even more for big models)
    :param bool repeat: one shot iterator or infinite generator
    :yield tuple: (img_batch, label_batch)
    """
    repeats = 2 ** 32 if repeat else 1
    random.seed(42)

    for i in range(repeats):

        if shuffle:
            random.shuffle(list(img_ann_list))  # random.shuffle(img_ann_list)

        for img_path, ann_path in img_ann_list:
            #just to see img name when checking img generator
            print(img_path)
            img_slices, labels = slice_and_label(img=img_path,
                                                 ann_path=ann_path,
                                                 roi=roi,
                                                 crop_size=crop_size,
                                                 resize=resize,
                                                 normalize=normalize)
            for i in range(0, img_slices.shape[0], batch_size):
                img_batch = img_slices[i:i + batch_size]
                label_batch = labels[i:i + batch_size]
                yield img_batch, label_batch


def train_val_image_generator(data_path, batch_size=128, crop_size=128, ext="png", normalize=True, resize=1.0,
                              roi=None, repeat=True):
    # load img and annotation filepath recursively from folder
    img_list = [img for img in sorted(glob.glob(data_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(data_path + "**/*." + "json", recursive=True))]

    # separate images/annotation by annotations
    normal_img_ann_list, defect_img_ann_list = separate_images_by_label(img_list, ann_list)

    # split to train/val by annotations (normal/defect)
    test_generator = crop_generator(defect_img_ann_list, roi=roi, batch_size=batch_size,
                                    crop_size=crop_size,
                                    normalize=normalize,
                                    resize=resize,
                                    repeat=repeat,
                                    shuffle=False)

    # if there's not need for validation (e.g. for GANs) return only train and test generators
    train_generator = crop_generator(normal_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                     normalize=normalize,
                                     roi=roi,
                                     repeat=repeat,
                                     shuffle=True,
                                     resize=resize)

    return train_generator, test_generator


def train_val_for_classifier(data_path, batch_size=128, crop_size=128, ext="png", roi=None,
                             normalize=True, resize=False,
                             val_frac=0.2, balanced=True):
    # load img and annotation filepath recursively from folder
    img_list = [img for img in sorted(glob.glob(data_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(data_path + "**/*." + "json", recursive=True))]

    # separate images/annotation by label
    normal_img_ann_list, defect_img_ann_list = separate_images_by_label(img_list, ann_list)

    # split to train/val
    random.seed(42)
    random.shuffle(defect_img_ann_list)
    num_images = len(defect_img_ann_list)
    train_val_split = int(num_images * (1 - val_frac))
    train_img_ann_list = defect_img_ann_list[:train_val_split]
    val_img_ann_list = defect_img_ann_list[train_val_split:]

    train_generator = crop_generator(train_img_ann_list,
                                     batch_size=999 if balanced else batch_size,
                                     crop_size=crop_size,
                                     roi=roi,
                                     normalize=normalize,
                                     repeat=False if balanced else True,
                                     shuffle=False,
                                     resize=resize)
    val_generator = crop_generator(val_img_ann_list,
                                   batch_size=999 if balanced else batch_size,
                                   crop_size=crop_size,
                                   roi=roi,
                                   normalize=normalize,
                                   repeat=False if balanced else True,
                                   shuffle=False,
                                   resize=resize)

    if balanced:  # load dataset to memory
        train_images, train_labels = gen_to_dataset(train_generator)
        val_images, val_labels = gen_to_dataset(val_generator)
        balanced_train_generator = balanced_batch_generator(train_images,
                                                            to_categorical(train_labels),
                                                            batch_size=batch_size)
        balanced_val_generator = balanced_batch_generator(val_images,
                                                          to_categorical(val_labels),
                                                          batch_size=batch_size)
        return balanced_train_generator, balanced_val_generator

    return train_generator, val_generator


def gen_to_dataset(gen):
    images, labels = zip(*list(gen))
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels


def balanced_batch_generator(x, y, batch_size, categorical=False):
    """A generator for creating balanced batched.
    This generator loops over its data indefinitely and yields balanced,
    shuffled batches.
    Args:
    x (numpy.ndarray): Samples (inputs). Must have the same length as `y`.
    y (numpy.ndarray): Labels (targets). Must be a binary class matrix (i.e.,
        shape `(num_samples, num_classes)`). You can use `keras.utils.to_categorical`
        to convert a class vector to a binary class matrix.
    batch_size (int): Batch size.
    categorical (bool, optional): If true, generates binary class matrices
        (i.e., shape `(num_samples, num_classes)`) for batch labels (targets).
        Otherwise, generates class vectors (i.e., shape `(num_samples, )`).
        Defaults to `True`.
    Returns a generator yielding batches as tuples `(inputs, targets)` that can
        be directly used with Keras.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Args `x` and `y` must have the same length.')
    if len(y.shape) != 2:
        raise ValueError(
            'Arg `y` must have a shape of (num_samples, num_classes). ' +
            'You can use `keras.utils.to_categorical` to convert a class vector ' +
            'to a binary class matrix.'
        )
    if batch_size < 1:
        raise ValueError('Arg `batch_size` must be a positive integer.')
    num_samples = y.shape[0]
    num_classes = y.shape[1]
    batch_x_shape = (batch_size, *x.shape[1:])
    batch_y_shape = (batch_size, num_classes) if categorical else (batch_size,)
    indexes = [0 for _ in range(num_classes)]
    samples = [[] for _ in range(num_classes)]
    for i in range(num_samples):
        samples[np.argmax(y[i])].append(x[i])
    while True:
        batch_x = np.ndarray(shape=batch_x_shape, dtype=x.dtype)
        batch_y = np.zeros(shape=batch_y_shape, dtype=y.dtype)
        for i in range(batch_size):
            random_class = random.randrange(num_classes)
            current_index = indexes[random_class]
            indexes[random_class] = (current_index + 1) % len(samples[random_class])
            if current_index == 0:
                random.shuffle(samples[random_class])
            batch_x[i] = samples[random_class][current_index]
            if categorical:
                batch_y[i][random_class] = 1
            else:
                batch_y[i] = random_class
        yield (batch_x, batch_y)


if __name__ == '__main__':
    cogwheel_type = '44H'
    data_path = f"C:/Users/3978/Desktop/44H/train/all"

    import os 
    os.chdir(f"C:/Users/3978/Desktop/Autoencoder/MAI_source_code/AE/dataloader")

    # fetch data parameters from config
    cogwheel_params = yaml.safe_load(open("cogwheel_config.yml"))[cogwheel_type]

    # # init data generators
    train_img_gen, val_img_gen = train_val_image_generator(data_path=data_path,
                                                           batch_size=999,
                                                           normalize=False,
                                                           **cogwheel_params
                                                           )
    for _ in range(100):
        img_batch, label = next(val_img_gen)
        # if we want to see how many crops
        print('All crops: {}'.format(len(img_batch)))
        print(np.where(label == 1)[0])
        display_slices(img_batch.astype(np.uint8),label)

    # init data generators
    # train_img_gen, val_img_gen = train_val_for_classifier(data_path=data_path,
    #                                                       batch_size=16,
    #                                                       val_frac=0.2,
    #                                                       normalize=False,
    #                                                       balanced=True,
    #                                                       **cogwheel_params
    #                                                       )
    
    # img_batch, label = next(train_img_gen)
    # print(np.where(label == 1)[0])
    # display_slices(img_batch.astype(np.uint8),label)
