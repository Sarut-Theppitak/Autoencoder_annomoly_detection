import yaml
from absl import app
from absl import flags

from autoencoder.training.ae_test import eval_scores, eval_contours
from dataloader.image_generators import train_val_image_generator
from eval_utils.metrics_vis import display_confusion_matrix

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "C:/Users/3978/Desktop/Autoencoder/MAI_source_code/data/test/bar", "absolute dir path for image dataset")
flags.DEFINE_integer("batch_size", 32, "Size of testing batches")
flags.DEFINE_string("cogwheel_type", "8GC", "Type of cogwheel that's present in the data")
flags.DEFINE_string("method", 'scores', "method for evaluating the autoencoder model")

# FLAGS for scores method
flags.DEFINE_string("score_type", 'psnr', "anomaly score type dssim / l2 / psnr")

# FLAGS for contour method
flags.DEFINE_boolean("debug", True, "Visualize anomaly detection pipeline for contour method")
flags.DEFINE_boolean("show_contours", True, "Display polygons around contours if detected")
flags.DEFINE_integer("diff_threshold", 25, "absolute difference threshold for difference map")
flags.DEFINE_float("min_percentile", 99.5, "minimum percentile value for difference map threshold")
flags.DEFINE_integer("min_area", 20, "minimum area for contour (in pixels) to be considered an anomaly")


def main(argv=None):
    # fetch data parameters from config
    data_path = f"{FLAGS.data_dir}/"
    cogwheel_params = yaml.safe_load(open("./dataloader/cogwheel_config.yml"))[FLAGS.cogwheel_type]

    # init data generators
    train_img_gen, test_img_gen = train_val_image_generator(data_path=data_path,
                                                            batch_size=FLAGS.batch_size,
                                                            normalize=True,
                                                            repeat=False,
                                                            **cogwheel_params
                                                                 )
    logs_dir = f'./checkpoints_skeleton/autoencoder/training/{FLAGS.cogwheel_type}'

    if FLAGS.method == 'scores':
        scores, labels = eval_scores(data_generator=test_img_gen,
                                     score_type=FLAGS.score_type,
                                     logs_dir=logs_dir)

    elif FLAGS.method == 'contours':
        predictions, labels = eval_contours(data_generator=test_img_gen,
                                            logs_dir=logs_dir,
                                            show_contours=FLAGS.show_contours,
                                            debug=FLAGS.debug,
                                            threshold=FLAGS.diff_threshold,
                                            min_percentile=FLAGS.min_percentile,
                                            min_area=FLAGS.min_area)

        display_confusion_matrix(predictions, labels)


if __name__ == '__main__':
    app.run(main)
