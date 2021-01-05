import glob
import cv2
import numpy as np
import tensorflow as tf
import yaml
import ntpath
import os
from tqdm import tqdm
from pathlib import Path
from dataloader.processing.slicing import slice_and_label, necessary_splits
from eval_utils.metrics_vis import plot_classification_report, display_confusion_matrix
from models.autoencoders import XAutoencoder
from models.bit import BiT
from models.xception import FineTunedXception

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


class Pipeline(object):
    def __init__(self,
                 model_type,
                 cogwheel_type,
                 ae_checkpoint_dir,
                 cls_checkpoint_dir,
                 batch_size=16,
                 error_analysis_threshold=False):

        # cogwheel params
        base_path = Path(__file__).parent
        config_path = (base_path / "../dataloader/cogwheel_config.yml").resolve()
        self.cogwheel_params = yaml.safe_load(open(config_path))[cogwheel_type]

        # hardware settings
        self.batch_size = batch_size

        # init AE model
        self.autoencoder = XAutoencoder()
        # load weights from checkpoint
        self.ae_checkpoint = tf.train.Checkpoint(autoencoder=self.autoencoder)
        self.ae_checkpoint.restore(tf.train.latest_checkpoint(ae_checkpoint_dir))

        # init classifier
        if model_type == 'xception':
            self.cls = FineTunedXception()
        else:
            self.cls = BiT(model_type, train_bit=False)

        # checkpoint writer
        self.cls_checkpoint = tf.train.Checkpoint(cls=self.cls)

        # restore from checkpoint if exists
        self.cls_checkpoint.restore(tf.train.latest_checkpoint(cls_checkpoint_dir))

        # perform error analysis on most prominent classification mistakes
        self.error_analysis_threshold = error_analysis_threshold

    def preprocess(self, tensor):
        def pipeline(self, img):
            tensor = tf.cast(img, tf.float32)

            # AE feed-forward
            x_hat = self.autoencoder(tensor, training=False)
            tensor_inv = -tensor
            x_hat_inv = self.autoencoder(tensor_inv, training=False)

            # compute difference maps
            diff_map = tf.abs(tensor - x_hat)
            diff_map_inv = tf.abs(tensor_inv - x_hat_inv)

            # make color image
            tensor = tf.concat([diff_map, diff_map_inv, tensor + 1], axis=-1)

            return tensor

        return pipeline(self, tensor)

    def max_score_from_slices(self, img_slices, labels, img_name, debug=False):
        highest_defect_score = 0.0
        all_ng_count = 0
        ng_number = 1
        fn_number = 1
        for i in range(0, img_slices.shape[0], self.batch_size):
            img_batch = img_slices[i:i + self.batch_size]
            img_colour_batch = self.preprocess(img_batch)
            scores = self.cls(img_colour_batch, training=False)
            max_score = tf.reduce_max(scores)
            ng_count = np.count_nonzero(np.where(np.array(scores) > self.error_analysis_threshold, 1.0, 0.0))

            if max_score > highest_defect_score:
                highest_defect_score = max_score
            all_ng_count += ng_count

            if debug:
                ng_indexs = np.where(np.array(scores) > self.error_analysis_threshold)[0]
                ok_index = np.where(np.array(scores) <= self.error_analysis_threshold)[0]
                for index in ng_indexs:
                    label = int(labels[index])
                    img = img_batch[index]
                    img = cv2.cvtColor(img.astype('float32') ,cv2.COLOR_GRAY2RGB)
                    img = ((img + 1) * 127.5).astype(np.uint8)
                    img_colour = img_colour_batch[index].numpy()
                    img_colour = cv2.cvtColor(img_colour, cv2.COLOR_RGB2BGR)
                    img_colour = (img_colour * 255).astype(np.uint8)
                    to_show =  cv2.hconcat([img, img_colour])
                    cv2.imshow('number:{}-score:{}-label:{}-{}'.format(ng_number,scores[index],label,img_name), to_show)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    ng_number += 1
                for index in ok_indexs:
                    label = int(labels[index])
                    if label == 1:
                        img = img_batch[index]
                        img = cv2.cvtColor(img.astype('float32') ,cv2.COLOR_GRAY2RGB)
                        img = ((img + 1) * 127.5).astype(np.uint8)
                        img_colour = img_colour_batch[index].numpy()
                        img_colour = cv2.cvtColor(img_colour, cv2.COLOR_RGB2BGR)
                        img_colour = (img_colour * 255).astype(np.uint8)
                        to_show =  cv2.hconcat([img, img_colour])
                        cv2.imshow('number:{}-score:{}-label:{}-{}'.format(ng_number,scores[index],label,img_name), to_show)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        fn_number += 1
        return highest_defect_score, all_ng_count

    def inference(self, img, localiaztion=False):
        img_slices, labels = slice_and_label(img, **self.cogwheel_params)

        highest_defect_score = self.max_score_from_slices(img_slices, labels,  img_name=img, debug=False)

        if not localiaztion:
            bboxes = [
                {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0], "score": highest_defect_score.numpy()}
            ]

        return bboxes

    def eval(self, img, ann_path=None, debug=False):
        img_slices, labels = slice_and_label(img, ann_path=ann_path, **self.cogwheel_params)
        image_level_label = np.max(labels)
        img_name = ntpath.split(img)[1]

        highest_defect_score, all_ng_count = self.max_score_from_slices(img_slices, labels, img_name, debug=debug)

        if image_level_label and (self.error_analysis_threshold > highest_defect_score) and isinstance(img, str):
            with open("./logs/error_analysis.txt", "a+") as f:
                f.write(f'{img}\n')

        return highest_defect_score, image_level_label, all_ng_count

    def highlight_defects(self, img_path):
        img = cv2.imread(img_path, 0)
        img_slices, labels = slice_and_label(img_path, ann_path=None, **self.cogwheel_params)
        colored_slices = self.preprocess(img_slices)

        width = img.shape[1]
        height = img.shape[0]
        crop_size = self.cogwheel_params['crop_size']

        min_height = self.cogwheel_params['roi']['ymin']
        max_height = height - self.cogwheel_params['roi']['ymax']

        v_splits = necessary_splits(width, crop_size)
        h_splits = necessary_splits(max_height - min_height, crop_size)

        v_range = np.linspace(start=0, stop=width - crop_size, num=v_splits, dtype=np.int32)
        h_range = np.linspace(start=min_height, stop=max_height - crop_size, num=h_splits, dtype=np.int32)

        reconstructed_image = np.zeros((height, width, 3))
        i = 0
        for vertical in v_range:
            for horizontal in h_range:
                reconstructed_image[horizontal:horizontal + crop_size, vertical:vertical + crop_size, :] = \
                    colored_slices[i]
                i += 1

        cv2.imshow('Defect Heatmap', reconstructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def eval_dir(self, data_dir, batch_size=16, ext="png", debug=False, display_method='confusion', eval_method='max_score', ng_count_threshold = 1):
        img_list = [img for img in sorted(glob.glob(data_dir + "**/*." + ext, recursive=True))]
        ann_list = [img for img in sorted(glob.glob(data_dir + "**/*." + "json", recursive=True))]

        defect_scores = []
        image_level_labels = []
        all_ng_counts = []
        with open("./logs/eval_results.txt", "w") as f:
            f.write(f'')
        for img_path, ann_path in tqdm(zip(img_list, ann_list)):
            defect_score, image_level_label, all_ng_count = self.eval(img_path, ann_path, debug=debug)
            defect_scores.append(defect_score)
            image_level_labels.append(image_level_label)
            all_ng_counts.append(all_ng_count)
            print(f'\nimages path: ', img_path)
            print('score : ', defect_score, ' || ', 'label: ', image_level_label, ' || ', 'ng_count: ', all_ng_count)
            with open("./logs/eval_results.txt", "a+") as f:
                f.write(f'images path: {img_path}\nscore : {defect_score} || label: {image_level_label} || ng_count: {all_ng_count}\n')
        
        all_ng_counts = np.array(all_ng_counts)
        defect_scores = np.array(defect_scores)
        image_level_labels = np.array(image_level_labels)

        if display_method == 'confusion':
            if eval_method == 'max_score':
                predictions = np.where(defect_scores > self.error_analysis_threshold, 1.0, 0.0)
            elif eval_method == 'ng_count':
                predictions = np.where(all_ng_counts > ng_count_threshold, 1.0, 0.0)
            display_confusion_matrix(predictions, image_level_labels)
        elif display_method == 'classification_report':
            plot_classification_report(defect_scores, image_level_labels)
        elif display_method == 'both':
            if eval_method == 'max_score':
                predictions = np.where(defect_scores > self.error_analysis_threshold, 1.0, 0.0)
            elif eval_method == 'ng_count':
                predictions = np.where(all_ng_counts > ng_count_threshold, 1.0, 0.0)
            display_confusion_matrix(predictions, image_level_labels)
            plot_classification_report(defect_scores, image_level_labels)

    def preprocess_encoder(self, tensor, score_type='l2'):
        tensor = tf.cast(img, tf.float32)

        # AE feed-forward
        x_hat = self.autoencoder(tensor, training=False)
        
        #compute loss
        if score_type == 'dssim':
            # reconstruction DSSIM
            anomaly_score = (1 - tf.image.ssim((tensor + 1) / 2, (x_hat + 1) / 2, max_val=1)) / 2

        elif score_type == 'l2':
            anomaly_score = tf.norm(tensor - x_hat, ord=2, axis=(1, 2))

        elif score_type == 'psnr':
            anomaly_score = tf.image.psnr((tensor + 1) / 2, (x_hat + 1) / 2, max_val=1)

        return x_hat,anomaly_score

    def max_score_from_slices_pai(self, img_slices, labels, img_name, debug=False, score_type='l2'):
        highest_defect_score = 0.0
        all_ng_count = 0
        ng_number = 1
        img_no_cls = []
        img_cls = []
        x_hat_no_cls = []
        x_hat_cls = []
        anomoly_score_threshold_max = 10
        anomoly_score_threshold_min = 5

        for i in range(0, img_slices.shape[0], self.batch_size):
            img_batch = img_slices[i:i + self.batch_size]
            x_hat, anomaly_score  = self.preprocess_encoder(img_batch, score_type=score_type)
            anomaly_scores = np.array(anomaly_scores)
            over_indices = np.where(anomaly_score > anomoly_score_threshold_max)
            under_indices = np.where(anomaly_score < anomoly_score_threshold_min)

            # try both method which one is faster 


            scores = self.cls(img_colour_batch, training=False)
            max_score = tf.reduce_max(scores)
            ng_count = np.count_nonzero(np.where(np.array(scores) > self.error_analysis_threshold, 1.0, 0.0))

            if max_score > highest_defect_score:
                highest_defect_score = max_score
            all_ng_count += ng_count

            if debug:
                ng_indexs = np.where(np.array(scores) > self.error_analysis_threshold)[0]
                for index in ng_indexs:
                    img = img_batch[index]
                    img = cv2.cvtColor(img.astype('float32') ,cv2.COLOR_GRAY2RGB)
                    img = ((img + 1) * 127.5).astype(np.uint8)
                    img_colour = img_colour_batch[index].numpy()
                    img_colour = cv2.cvtColor(img_colour, cv2.COLOR_RGB2BGR)
                    img_colour = (img_colour * 255).astype(np.uint8)
                    to_show =  cv2.hconcat([img, img_colour])
                    cv2.imshow('number:{}-score:{}-{}'.format(ng_number,scores[index],img_name), to_show)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    ng_number += 1

        return highest_defect_score, all_ng_count            

    def eval_pai(self, img, ann_path=None, debug=False, score_type='l2'):
        # if no crop goes to cls highest_defect_score = 0 or 1  depends on l2 threshold
        img_slices, labels = slice_and_label(img, ann_path=ann_path, **self.cogwheel_params)
        image_level_label = np.max(labels)
        img_name = ntpath.split(img)[1]

        highest_defect_score, all_ng_count = self.max_score_from_slices_pai(img_slices, labels, img_name, debug=debug, score_type=score_type)

        if image_level_label and (self.error_analysis_threshold > highest_defect_score) and isinstance(img, str):
            with open("./logs/error_analysis.txt", "a+") as f:
                f.write(f'{img}\n')

        return highest_defect_score, image_level_label, all_ng_count
    
    def eval_main_pai(self, data_dir, batch_size=16, ext="png", debug=False, display_method='confusion', eval_method='max_score', ng_count_threshold = 1, score_type='l2'):
        img_list = [img for img in sorted(glob.glob(data_dir + "**/*." + ext, recursive=True))]
        ann_list = [img for img in sorted(glob.glob(data_dir + "**/*." + "json", recursive=True))]

        defect_scores = []
        image_level_labels = []
        all_ng_counts = []

        with open("./logs/eval_results.txt", "w") as f:
            f.write(f'')

        for img_path, ann_path in tqdm(zip(img_list, ann_list)):
            defect_score, image_level_label, all_ng_count = self.eval_pai(img_path, ann_path, debug=debug, score_type=score_type)

