import os

import matplotlib
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Progbar
from dataloader.processing.augmentations import augment
from eval_utils.tensorboard_utils import figure_to_tf_image, confusion_matrix_figure, pr_curve_figure, roc_figure
from models.autoencoders import XAutoencoder
from models.xception import FineTunedXception
from models.bit import BiT

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


class SSTrainer(object):
    def __init__(self,
                 model_type,
                 train_backbone,
                 train_generator,
                 val_generator,
                 training_steps,
                 reset_optimizer,
                 lr,
                 do_augmentations,
                 ae_checkpoint_dir,
                 save_checkpoint_every_n_steps,
                 logs_dir,
                 debug):

        # data generators
        self.train_generator = train_generator
        self.val_generator = val_generator

        # init AAE model
        self.autoencoder = XAutoencoder()
        # load weights from checkpoint
        self.ae_checkpoint = tf.train.Checkpoint(autoencoder=self.autoencoder)
        self.ae_checkpoint.restore(tf.train.latest_checkpoint(ae_checkpoint_dir))

        # init classifier
        if model_type == 'xception':
            self.cls = FineTunedXception()
        else:
            self.cls = BiT(model_type, train_bit=train_backbone)

        # optimizers
        self.training_steps = training_steps
        boundaries = [int(training_steps * 0.3), int(training_steps * 0.6), int(training_steps * 0.9)]
        self.lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=[lr, lr * 0.1, lr * 0.01, lr * 0.001])
        self.cls_optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule,
                                                     momentum=0.9,
                                                     clipvalue=0.5,
                                                     name='bit_optimizer')

        # augmentations
        self.do_augmentations = do_augmentations

        # checkpoint writer
        self.logs_dir = logs_dir
        self.checkpoint_dir = self.logs_dir + '/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.cls_checkpoint = tf.train.Checkpoint(cls=self.cls,
                                                  cls_optimizer=self.cls_optimizer)
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps

        # restore from checkpoint if exists
        self.cls_checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        if reset_optimizer:
            for var in self.cls_optimizer.variables():
                var.assign(tf.zeros_like(var))

        # summary writers
        self.train_writer = tf.summary.create_file_writer(logs_dir + '/train', name='train_writer')
        self.val_writer = tf.summary.create_file_writer(logs_dir + '/validation', name='val_writer')
        self.image_writer = tf.summary.create_file_writer(logs_dir + '/images', name='image_writer')
        self.debug = debug
        if debug:
            self.debug_writer = tf.summary.create_file_writer(logs_dir + '/debug', name='debug_writer')

    def preprocess(self, tensor):
        def pipeline(self, img):
            tensor = tf.cast(img, tf.float32)

            # augment
            if self.do_augmentations:
                tensor = augment(tensor)

            # AE feed-forward
            x_hat = self.autoencoder(tensor, training=False)
            tensor_inv = -tensor
            x_hat_inv = self.autoencoder(tensor_inv, training=False)

            # compute difference maps
            diff_map = tf.abs(tensor - x_hat)
            diff_map_inv = tf.abs(tensor_inv - x_hat_inv)

            # make color image
            tensor = tf.concat([diff_map, diff_map_inv, tensor + 1], axis=-1)

            step = self.cls_optimizer.iterations.numpy()
            if self.debug and (step % 100):
                with self.debug_writer.as_default():
                    # [-1, 1] -> [0, 255]
                    orig_display = tf.image.grayscale_to_rgb(tf.cast((img + 1) * 127.5, tf.uint8))
                    diff_display = tf.image.grayscale_to_rgb(tf.cast((diff_map + 1) * 127.5, tf.uint8))
                    orig_inv_display = tf.image.grayscale_to_rgb(tf.cast((tensor_inv + 1) * 127.5, tf.uint8))
                    diff_inv_display = tf.image.grayscale_to_rgb(tf.cast((diff_map_inv + 1) * 127.5, tf.uint8))
                    colored = tf.cast(tensor * 255, tf.uint8)

                    concatenated_img = tf.concat([orig_display, diff_display,
                                                  orig_inv_display, diff_inv_display,
                                                  colored],
                                                 axis=2)
                    tf.summary.image('Original | Diff Map | Inverted |  Diff Map  |  Colored',
                                     concatenated_img,
                                     step=step,
                                     max_outputs=16)

            return tensor

        return pipeline(self, tensor)

    def train(self, validation_freq=10, validation_steps=100, score_threshold=0.5):
        """
        Trainer function for the seconds stage following the AE model, all function parameters explanations are detailed in the
        cls_train_main.py flags.
        The function flow is as follows:
        1. initializes the model, optimizers, tensorboard & checkpoints writers
        2. restores from checkpoint if exists
        3. Training loop:
            - send img batch through trained aae to get its outputs (encodings, reconstructions)
            - compute difference map between image and reconstructed image
            - train RelationNet on these images
            compute losses:
                - MSE loss
            - take optimizer step
            - write to tensorboard / save checkpoint every n steps
        """
        # progress bar
        progbar = Progbar(self.training_steps)

        for step in range(self.training_steps):
            progbar.update(step)

            # grab batch
            img_batch, labels = next(self.train_generator)
            img_batch = self.preprocess(img_batch)

            with tf.GradientTape() as tape:
                # input to BiT model
                logits = self.cls(img_batch, training=True)
                # compute classification loss
                train_loss = tf.reduce_mean(
                    self.weighted_binary_crossentropy(logits, labels))

            # compute gradients
            grad = tape.gradient(train_loss, self.cls.trainable_variables)

            # apply gradients
            self.cls_optimizer.apply_gradients(zip(grad, self.cls.trainable_variables))


            # save checkpoint
            step = self.cls_optimizer.iterations.numpy()
            if step % self.save_checkpoint_every_n_steps == 0:
                self.cls_checkpoint.save(file_prefix=self.checkpoint_prefix)

            # write summaries

            if step % validation_freq == 0:
                with self.train_writer.as_default():
                    tf.summary.scalar("crossentropy_loss", train_loss, step=step)
                    tf.summary.scalar("learning_rate", self.lr_schedule(step), step=step)

                # validation
                print(f'   Step {step} | Training loss: {train_loss}')
                print('Starting eval phase...')
                self.eval(training_step=step, validation_steps=validation_steps, score_threshold=score_threshold)
                print('Resuming training...')

    def eval(self, training_step, validation_steps=10, score_threshold=0.5):
        targets = []
        losses = []
        scores = []
        for val_step in range(validation_steps):
            # grab batch
            img_batch, labels = next(self.val_generator)
            img_batch = self.preprocess(img_batch)

            # input to classifier model
            logits = self.cls(img_batch, training=False)

            # accumulate important scalars
            losses.extend(self.weighted_binary_crossentropy(logits, labels))
            targets.extend(labels)
            scores.extend(logits)

        # compute mean loss over all batches
        losses = tf.concat(losses, axis=0)
        scores = tf.concat(scores, axis=0)
        val_loss = tf.reduce_mean(losses)

        # write summaries to tensorboard
        with tf.name_scope('validation_summaries') as scope:
            with self.val_writer.as_default():
                tf.summary.scalar("crossentropy_loss", val_loss, step=training_step)
                scores = tf.concat(scores, axis=0)
                targets = tf.reshape(tf.convert_to_tensor(targets, dtype=tf.float32), (-1, 1))
                predictions = tf.reshape(tf.cast((scores > score_threshold), tf.float32), (-1, 1))

                # general metrics for tensorboard - accuracy, recall, precision
                TP = tf.reduce_sum(tf.cast((predictions == 1), tf.float32) * tf.cast((targets == 1), tf.float32))
                TN = tf.reduce_sum(tf.cast((predictions == 0), tf.float32) * tf.cast((targets == 0), tf.float32))
                FP = tf.reduce_sum(tf.cast((predictions == 1), tf.float32) * tf.cast((targets == 0), tf.float32))
                FN = tf.reduce_sum(tf.cast((predictions == 0), tf.float32) * tf.cast((targets == 1), tf.float32))
                accuracy = (TN + TP) / (TN + TP + FP + FN)
                recall = TP / (TP + TN)
                precision = TP / (TP + FP)

                tf.summary.scalar("accuracy", accuracy, step=training_step)
                tf.summary.scalar("recall", recall, step=training_step)
                tf.summary.scalar("precision", precision, step=training_step)

            with self.image_writer.as_default():
                # Precision-Recall curve for TensorBoard
                prc_figure = pr_curve_figure(targets, scores)
                prc_image = figure_to_tf_image(prc_figure)
                tf.summary.image("Precision-Recall Curve", prc_image, step=training_step)

                # ROC curve
                roc_fig = roc_figure(targets, scores)
                roc_image = figure_to_tf_image(roc_fig)
                tf.summary.image("ROC Curve", roc_image, step=training_step)

                # Confusion Matrix
                # calculate the confusion matrix.
                cm = confusion_matrix(targets, predictions)
                # Log the confusion matrix as an image summary.
                cm_figure = confusion_matrix_figure(cm, class_names=['Normal', 'Anomaly'])
                cm_image = figure_to_tf_image(cm_figure)
                tf.summary.image("Confusion Matrix", cm_image, step=training_step)

            print(f'Step {training_step} | Validation loss: {val_loss}')

    @staticmethod
    def weighted_binary_crossentropy(logits, labels, positive_weight=1):
        labels = tf.cast(labels[:, None], dtype='float32')
        return -(positive_weight * labels * tf.math.log(logits) + (1-labels) * tf.math.log(1-logits))

    @staticmethod
    def colorize(grayscale, vmin=None, vmax=None, cmap=None):
            """
            A utility function for TensorFlow that maps a grayscale image to a matplotlib
            colormap for use with TensorBoard image summaries.
            Arguments:
              - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
                [height, width, 1].
              - vmin: the minimum value of the range used for normalization.
                (Default: value minimum)
              - vmax: the maximum value of the range used for normalization.
                (Default: value maximum)
              - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
                (Default: 'gray')
            Example usage:
            ```
            output = tf.random_uniform(shape=[256, 256, 1])
            output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='plasma')
            tf.summary.image('output', output_color)
            ```

            Returns a 3D tensor of shape [height, width, 3].
            """

            # normalize
            vmin = tf.reduce_min(grayscale) if vmin is None else vmin
            vmax = tf.reduce_max(grayscale) if vmax is None else vmax
            grayscale = (grayscale - vmin) / (vmax - vmin) # vmin..vmax

            # gather
            color_mapper = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
            colored = color_mapper(grayscale[:, :, :, 0])[:, :, :, :3]   # slicing dealing with extra channels
            colored = tf.cast(colored, tf.float32)
            return colored
