import yaml
from absl import app
from absl import flags

from second_stage.training.cls_trainer import SSTrainer
from dataloader.image_generators import train_val_for_classifier

FLAGS = flags.FLAGS

# data flags
flags.DEFINE_string("data_dir", "C:/Users/3978/Desktop/Autoencoder/MAI_source_code/data/train/bar", "absolute dir path for image dataset")
flags.DEFINE_string("cogwheel_type", "8GC", "Type of cogwheel that's present in the data")
flags.DEFINE_integer("batch_size", 8, "number of image crops per iteration")

# training flags
flags.DEFINE_string("model_type", "r101x1", "model type for transfer learning")
flags.DEFINE_boolean("train_backbone", True, "Whether to train the CNN backbone")
flags.DEFINE_integer("schedule_length", 500, "Base schedule length for fine-tuning")
flags.DEFINE_float("base_lr", 0.003, "Learning rate for optimizers")
flags.DEFINE_boolean("augment", False, "Whether images will be augmented during training")
flags.DEFINE_integer("save_checkpoint_every_n_steps", 1000, "Frequency for saving model checkpoints")
flags.DEFINE_boolean("debug", True, "Write debug summaries to TensorBoard (e.g. reconstruction maps)")
flags.DEFINE_boolean("reset_optimizer", True,
                     "Reset optimizer internal timestep for re-training or to continue from previous session")


# eval flags
flags.DEFINE_float("val_frac", 0.2, "fraction of the data to set aside for validation")
flags.DEFINE_float("score_threshold", 0.5, "Threshold for predicting anomalies from score")
flags.DEFINE_integer("validation_freq", 100, "Perform eval on validation set every n steps")
flags.DEFINE_integer("validation_steps", 10, "Number of validation steps (batches from generator)")


def main(argv=None):
    # fetch data parameters from config
    data_path = f"{FLAGS.data_dir}/"
    cogwheel_params = yaml.safe_load(open("./dataloader/cogwheel_config.yml"))[FLAGS.cogwheel_type]

    # init train and validation datasets
    train_img_gen, val_img_gen = train_val_for_classifier(data_path=data_path,
                                                          batch_size=FLAGS.batch_size,
                                                          normalize=True,
                                                          balanced=True,
                                                          val_frac=FLAGS.val_frac,
                                                          **cogwheel_params)

    # init seconds stage trainer
    ae_checkpoint_dir = f'./checkpoints_skeleton/autoencoder/training/{FLAGS.cogwheel_type}/checkpoints'
    logs_dir = f'checkpoints_skeleton/second_stage/training/{FLAGS.cogwheel_type}/{FLAGS.model_type}'

    # BiT HyperRule Parameters for small dataset
    lr = FLAGS.base_lr * FLAGS.batch_size / 512
    training_steps = int(FLAGS.schedule_length * 512 / FLAGS.batch_size)

    # init Second Stage Trainer
    trainer = SSTrainer(model_type=FLAGS.model_type,
                        train_backbone=FLAGS.train_backbone,
                        train_generator=train_img_gen,
                        val_generator=val_img_gen,
                        lr=lr,
                        do_augmentations=FLAGS.augment,
                        training_steps=training_steps,
                        reset_optimizer=FLAGS.reset_optimizer,
                        save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps,
                        ae_checkpoint_dir=ae_checkpoint_dir,
                        logs_dir=logs_dir,
                        debug=FLAGS.debug)

    # starting training (this will also perform eval steps by configuration)
    trainer.train(
        validation_freq=FLAGS.validation_freq,
        validation_steps=FLAGS.validation_steps)

if __name__ == '__main__':
    app.run(main)
