import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tqdm import tqdm

from eval_utils.metrics_vis import plot_classification_report
from models.autoencoders import XAutoencoder

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def eval_scores(data_generator,
                score_type,
                logs_dir):
    """
    Evaluation step for the autoencoder model. Initializes model and restores from checkpoint, loops over labeled test
    data, computes the reconstruction L2 loss for input data and normalizes scores to be from [0, 1].
    Displays PR curve for anomaly scores vs. labels.
    """

    # init AE model
    autoencoder = XAutoencoder()

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint = tf.train.Checkpoint(autoencoder=autoencoder)

    # restore from checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    labels = []
    anomaly_scores = []
    for img_batch, label_batch in tqdm(data_generator):
        img_batch = tf.cast(img_batch, tf.float32)
        x_hat = autoencoder(img_batch, training=False)

        if score_type == 'dssim':
            # reconstruction DSSIM
            anomaly_score = (1 - tf.image.ssim((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1)) / 2

        elif score_type == 'l2':
            anomaly_score = tf.norm(img_batch - x_hat, ord=2, axis=(1, 2))

        elif score_type == 'psnr':
            anomaly_score = tf.image.psnr((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1)

        labels.extend(label_batch)
        anomaly_scores.extend(anomaly_score)

    anomaly_scores = np.array(anomaly_scores)
    labels = np.array(labels)
    plot_classification_report(anomaly_scores, labels)

    return anomaly_scores, labels


def eval_contours(data_generator,
                  logs_dir,
                  show_contours,
                  debug,
                  threshold,
                  min_percentile,
                  min_area):
    """
    Evaluation step for the autoencoder model. Initializes model and restores from checkpoint, loops over labeled test
    data, finds contours from difference map (x - x_hat)
    """

    # init autoencoder model
    autoencoder = XAutoencoder()

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint = tf.train.Checkpoint(autoencoder=autoencoder)
    # restore from checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    labels = []
    predictions = []
    for img_batch, label_batch in tqdm(data_generator):
        # get reconstructed image
        x_hat = autoencoder(img_batch, training=False)

        # subtract reconstructed images from originals
        diff_maps = tf.abs(img_batch - x_hat) - 1  # because [-1, 1]

        # transform back to original values and format
        diff_maps = tf.cast((diff_maps + 1) * 127.5, tf.uint8)
        img_batch = tf.cast((img_batch + 1) * 127.5, tf.uint8)
        rec_imgs = tf.cast((x_hat + 1) * 127.5, tf.uint8)

        for orig_img, reconstructed, diff_map, label in zip(img_batch, rec_imgs, diff_maps, label_batch):
            pred = detect_anomalies_in_diff_map(diff_map, threshold, min_percentile, min_area,
                                                show_contours, debug, label, orig_img, reconstructed)
            predictions.append(pred)
            labels.append(label)

    return predictions, labels


def detect_anomalies_in_diff_map(diff_map, threshold, min_percentile, min_area, show_contours=True, debug=False,
                                 label=None, orig_img=None, reconstructed=None):
    # clean diff map from noise
    grey_opening = ndimage.grey_opening(diff_map[:, :, 0], (3, 3), mode='nearest')
    grey_opening = cv2.medianBlur(grey_opening, 3)

    # create masks for min value and min percentile value
    _, mask1 = cv2.threshold(grey_opening, threshold, 255, cv2.THRESH_BINARY)
    img_percentile = np.percentile(grey_opening, min_percentile)
    _, mask2 = cv2.threshold(grey_opening, img_percentile, 255, cv2.THRESH_BINARY)
    masks = mask1 * mask2

    # try to connect close dots
    kernel = np.ones((3, 3))
    thresh_img = cv2.morphologyEx(masks, cv2.MORPH_OPEN, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    final_map = np.zeros_like(thresh_img)
    b = 5  # border from edge
    final_map[b:-b, b:-b] = thresh_img[b:-b, b:-b]  # get rid of edge differences (many FP)
    contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_area = [cv2.contourArea(contour) for contour in contours]
    # filter by contour area
    contours = [contour for contour, area in zip(contours, contours_area) if area >= min_area]

    if debug and (orig_img is not None) and (label == 1):
        crop_size = orig_img.shape[1]
        print('Contours area: ', contours_area)
        panels = np.zeros((crop_size, crop_size * 6, 1))
        panels[:, :crop_size, :] = orig_img
        panels[:, crop_size:crop_size * 2, :] = reconstructed
        panels[:, crop_size * 2:crop_size * 3, :] = diff_map
        panels[:, crop_size * 3:crop_size * 4, :] = np.expand_dims(grey_opening, -1)
        panels[:, crop_size * 4:crop_size * 5, :] = np.expand_dims(masks * 255, -1)
        panels[:, crop_size * 5:crop_size * 6, :] = np.expand_dims(final_map * 255, -1)

        cv2.imshow(' Image  |  Reconstructed  |  Diff Map  |  Grey Open  |  Masks  |  Final Map  |  Label - {}'.format(
            str(label)),
            panels.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    found_contours = (len(contours) > 0)
    if show_contours and (orig_img is not None) and found_contours:
        crop_for_display = orig_img.numpy().astype(np.uint8)
        cv2.drawContours(crop_for_display, contours, -1, (255, 0, 0), 3)
        cv2.imshow(f'GT Label - {}'.format(str(label)), crop_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 1.0 if found_contours else 0.0
