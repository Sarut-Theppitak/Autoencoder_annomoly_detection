import cv2
import numpy as np


def show_heatmap(inputs, diff_map, crop_size, labels=None):
    for i in range(inputs.shape[0]):
        comparison = np.zeros((crop_size, crop_size, 3))
        comparison[:, :crop_size, :] = inputs[i]
        overlay = (inputs[i] * 0.7) + (diff_map[i] * 0.3)
        comparison[:, crop_size:crop_size * 2:, :] = cv2.applyColorMap(overlay.numpy().astype(np.uint8),
                                                                       cv2.COLORMAP_JET)
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Difference Heatmap     |     Label - {label}',
                   comparison.astype(np.uint8))
        cv2.waitKey(0)


def show_triptych(inputs, reconstructed, diff_map, crop_size, labels=None):
    for i in range(inputs.shape[0]):
        triptych = np.zeros((crop_size, int(crop_size * 3), 1))
        triptych[:, :crop_size, :] = inputs[i]
        triptych[:, crop_size:crop_size * 2, :] = reconstructed[i]
        triptych[:, crop_size * 2:crop_size * 3, :] = diff_map[i]
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Reconstructed    |     Difference      |     Label - {label}',
                   triptych.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
