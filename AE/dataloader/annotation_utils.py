import json

import numpy as np

def separate_images_by_label(img_list, ann_list):
    """
    filter globs of img list and ann list by whether they have bboxes in their annotation.json file
    :param list img_list, ann_list: lists of string datapaths from glob.glob
    :return tuple: normal_img_list (no bboxes), defect_img_list (bboxes of defects)
    """
    normal_img_list = []
    defect_img_list = []
    for img_path, ann_path, in zip(img_list, ann_list):
        ann_parsed = load_annotation_file(ann_path)
        bboxes = annotation_to_bboxes_ltwh(ann_parsed)
        if len(bboxes) == 0:
            normal_img_list.append((img_path, ann_path))
        elif len(bboxes) > 0:
            defect_img_list.append((img_path, ann_path))

    return normal_img_list, defect_img_list

def load_annotation_file(ann_path):
    """
    Loads a supervise.ly annotation file.

    :param str ann_path: Full path to the annotation file.
    :return dict: The parsed content of the annotation.
    """
    with open(ann_path) as f:
        return json.load(f)


def annotation_to_bboxes_ltwh(ann, classes='defect'):
    """
    Returns a (y, x, width, height) bbox for each object in the annotation.

    :param dict ann: An image annotation given in supervise.ly format
    :param tuple[str]|list[str]|None classes: The object classes to be extracted (e.g. 'defect' or 'roi').
                                              None means any.
    :rtype List[(int, int, int, int)]
    """

    def obj_to_bbox(obj):
        p = np.array(obj['points']['exterior'])
        mn = np.min(p, axis=0)
        mx = np.max(p, axis=0)
        return tuple(mn) + tuple(mx - mn)

    def filter_class(obj):
        return obj['classTitle'] in classes if classes is not None else True

    return list(map(obj_to_bbox, filter(filter_class, ann['objects'])))
