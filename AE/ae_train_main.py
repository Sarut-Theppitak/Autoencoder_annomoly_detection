import yaml
from absl import app
from absl import flags

from autoencoder.training.ae_trainer import Trainer
from dataloader.image_generators import train_val_image_generator

FLAGS = flags.FLAGS

# data flags
flags.DEFINE_string("data_dir", "C:/Users/3978/Desktop/Autoencoder/MAI_source_code/data/train/bar", "absolute dir path for image dataset")
flags.DEFINE_string("cogwheel_type", "8GC", "Type of cogwheel that's present in the data")
flags.DEFINE_integer("batch_size", 2, "Size of training batches")

# training flags
flags.DEFINE_integer("training_steps", 100001, "Number of training steps")
flags.DEFINE_float("initial_lr", 0.0009, "Initial learning rate base for optimizer (will be decayed by policy)")
flags.DEFINE_list("loss_weights", [0.16, 0.84], "weights for L1/L2 reconstruction, dssim loss")
flags.DEFINE_integer("norm_ord", 1, "order for loss norm (L1/L2")
flags.DEFINE_float("activity_regularizer", 1e-6, "Activity regularization on the bottleneck feature vector")
flags.DEFINE_boolean("augment", True, "Whether images will be augmented during training (in addition to noise)")
flags.DEFINE_integer("save_checkpoint_every_n_steps", 10000, "Frequency for saving model checkpoints")
flags.DEFINE_boolean("reset_optimizer", True,
                     "Reset optimizer internal timestep for re-training or to continue from previous session")

# eval flags
flags.DEFINE_integer("validation_freq", 2000, "Perform eval on validation set every n steps")
flags.DEFINE_integer("validation_steps", 30, "Number of validation steps (batches from generator)")


def main(argv=None):
    # fetch data parameters from config
    data_path = f"{FLAGS.data_dir}"
    cogwheel_params = yaml.safe_load(open("./dataloader/cogwheel_config.yml"))[FLAGS.cogwheel_type]

    # init data generators
    train_img_gen, test_img_gen = train_val_image_generator(data_path=data_path,
                                                            batch_size=FLAGS.batch_size,
                                                            normalize=True,
                                                            **cogwheel_params
                                                            )

    # init model trainer
    logs_dir = f'checkpoints_skeleton/autoencoder/training/{FLAGS.cogwheel_type}'
    trainer = Trainer(train_generator=train_img_gen,
                      val_generator=test_img_gen,
                      training_steps=FLAGS.training_steps,
                      reset_optimizer=FLAGS.reset_optimizer,
                      initial_lr=FLAGS.initial_lr,
                      activity_regularizer=FLAGS.activity_regularizer,
                      logs_dir=logs_dir)

    trainer.train(
        do_augmentations=FLAGS.augment,
        loss_weights=FLAGS.loss_weights,
        norm_ord=FLAGS.norm_ord,
        validation_freq=FLAGS.validation_freq,
        validation_steps=FLAGS.validation_steps,
        save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps)


if __name__ == '__main__':
    app.run(main)
