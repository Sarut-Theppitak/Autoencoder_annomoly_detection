import cv2
import numpy as np

from dataloader.annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh
from dataloader.processing.preprocessing import stack_and_expand, center_and_scale
# from annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh
# from processing.preprocessing import stack_and_expand, center_and_scale


def slice_and_label(img, ann_path=None, crop_size=128, resize=1.0, roi=None, normalize=True):
    if isinstance(img, str):
        img = cv2.imread(img, 0)

    if normalize:
        img = center_and_scale(img)

    if ann_path is not None:
        ann = load_annotation_file(ann_path)
        bboxes = annotation_to_bboxes_ltwh(ann)
    else:
        bboxes = None

    img_slices, labels = img_slice_and_label(img=img,
                                             crop_size=crop_size,
                                             bboxes=bboxes,
                                             roi=roi,
                                             resize=resize)

    return img_slices, labels


def display_slices(img_slices,label):
    """ simple function to show the result of your img_slices from function img_slice_and_labels"""
    for i, img in enumerate(img_slices):
        if i in np.where(label == 1)[0]:
            cv2.imshow('Image Slice ' + str(i), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def bboxes_included_in_crop(vertical, horizontal, interval, bboxes, partial=True):
    """
    Check whether there's a bbox inside the crop
    :param int vertical: y coordinate
    :param int horizontal: x coordinate
    :param int interval: size of height or width from coordinates
    :param list bboxes: list of bboxes in (y, x, width, height) format
    :param bool partial: whether partial bboxes inside crops should be counted as included
    :return float: 1.0 if crop contains bbox else 0.0
    """
    for y, x, w, h in bboxes:
        if partial:
            percent_w = w * 0
            percent_h = h * 0
            cond_1 = (vertical <= y + percent_w) and (y + percent_w <= vertical + interval)
            cond_2 = (horizontal <= x + percent_h) and (x + percent_h <= horizontal + interval)
            cond_3 = (vertical <= y + w - percent_w) and (y + w - percent_w <= vertical + interval)
            cond_4 = (horizontal <= x + h - percent_h) and (x + h - percent_h <= horizontal + interval)
            if all([cond_1, cond_2]) or all([cond_3, cond_4]):
                return 1.0

        else:
            cond_1 = (vertical <= y) and (y + w <= vertical + interval)
            cond_2 = (horizontal <= x) and (x + h <= horizontal + interval)
            if all([cond_1, cond_2]):
                return 1.0

    return 0.0


def img_slice_and_label(img, crop_size, bboxes=None, roi=None, resize=1.0):
    """
    Takes an image and slices it to squares of crop_size x crop_size
    Can additionally resize the crops (usually to shrink to smaller img than crop_size)
    if bboxes from annotations are available, can label the crops 1 if bbox is present or 0 if not
    :param np.array img: array of image
    :param int crop_size: crop_size by which to slice
    :param list bboxes: list bboxes in (y, x, width, height) format
    :param bool or float resize: resize the slices after cropping (usually to shrink even more for big models)
    :return tuple (img_slices, labels): tuple of 2 lists img_slices and labels (all 0.0 if no bboxes)
    """
    width = img.shape[1]
    height = img.shape[0]
    img_slices = []
    labels = []

    min_height = roi['ymin']
    max_height = height - roi['ymax']

    v_splits = necessary_splits(width, crop_size)
    h_splits = necessary_splits(max_height - min_height, crop_size)

    v_range = np.linspace(start=0, stop=width-crop_size, num=v_splits, dtype=np.int32)
    h_range = np.linspace(start=min_height, stop=max_height-crop_size, num=h_splits, dtype=np.int32)

    for vertical in v_range:
        for horizontal in h_range:
            crop = img[horizontal:horizontal + crop_size, vertical:vertical + crop_size]

            if resize:
                w = int(crop.shape[1] * resize)
                h = int(crop.shape[0] * resize)
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

            img_slices.append(crop)
            if bboxes:
                label = bboxes_included_in_crop(vertical, horizontal, crop_size, bboxes)
                labels.append(label)
            else:
                labels.append(0.0)

    img_slices = stack_and_expand(img_slices)
    labels = np.array(labels)
    return img_slices, labels


def necessary_splits(axis, crop_size):
    quotient, remainder = divmod(axis, crop_size)
    return quotient + 1
