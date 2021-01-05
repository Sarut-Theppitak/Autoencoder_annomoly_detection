from absl import app
from absl import flags

from pipeline.inference import Pipeline

FLAGS = flags.FLAGS

# data flags
flags.DEFINE_string("data_dir", "C:/Users/3978/Desktop/Autoencoder/MAI_source_code/data/test/bar", "absolute dir path for image dataset")
flags.DEFINE_string("cogwheel_type", "8GC", "Type of cogwheel that's present in the data")

# model flags
flags.DEFINE_string("model_type", "r101x1", "model type for transfer learning")
flags.DEFINE_integer("batch_size", 128, "number of image crops per iteration, depends on hardware capabilities. the larger, the faster")

# debug flags
flags.DEFINE_float("error_analysis_threshold", 0.50,
                   "score threshold to log image path of false negative")
flags.DEFINE_boolean("debug_flag", False, "flag to visualize the ng crops")

# display_method
flags.DEFINE_string("display_method", "confusion", "confusion / classification_report / both")

# flag for confusion matrix method
flags.DEFINE_string("eval_method", "max_score", "max_score or ng_count")

# flag for confusion matrix-ng_count 
flags.DEFINE_integer("ng_count_threshold", 1, "the threshold of ng_counts value for image to be considered as ng if the real ng_counts is more than the threshold")


def main(argv=None):
    # fetch data parameters from config
    data_path = f"{FLAGS.data_dir}/"

    # grab checkpoints from checkpoint structure
    ae_checkpoint_dir = f'./checkpoints_skeleton/autoencoder/training/{FLAGS.cogwheel_type}/checkpoints'
    cls_checkpoint_dir = f'./checkpoints_skeleton/second_stage/training/{FLAGS.cogwheel_type}/{FLAGS.model_type}/checkpoints'

    # init pipeline
    pipeline = Pipeline(model_type=FLAGS.model_type,
                        cogwheel_type=FLAGS.cogwheel_type,
                        ae_checkpoint_dir=ae_checkpoint_dir,
                        cls_checkpoint_dir=cls_checkpoint_dir,
                        error_analysis_threshold=FLAGS.error_analysis_threshold)

    # start eval on data dir
    pipeline.eval_dir(data_path, 
                      batch_size=FLAGS.batch_size, 
                      debug=FLAGS.debug_flag,
                      display_method=FLAGS.display_method,
                      eval_method=FLAGS.eval_method,
                      ng_count_threshold=FLAGS.ng_count_threshold)


if __name__ == '__main__':
    app.run(main)
