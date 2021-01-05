import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar

from dataloader.processing.augmentations import augment
from models.autoencoders import XAutoencoder

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


class Trainer(object):
    def __init__(self,
                 train_generator,
                 val_generator,
                 training_steps,
                 reset_optimizer,
                 initial_lr,
                 activity_regularizer,
                 logs_dir):

        """    1. initializes the model, optimizers, tensorboard & checkpoints writers
               2. restores from checkpoint if exists
        """
        # data generators
        self.train_generator = train_generator
        self.val_generator = val_generator

        # init autoencoder model
        self.autoencoder = XAutoencoder(activity_regularizer)

        # init autoencoder model and optimizer
        self.training_steps = training_steps
        self.lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[int(training_steps * 0.3), int(training_steps * 0.6), int(training_steps * 0.9)],
            values=[initial_lr, initial_lr * 0.1, initial_lr * 0.01, initial_lr * 0.001])
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.9, clipvalue=0.5)

        # summary writers
        self.train_writer = tf.summary.create_file_writer(logs_dir + '/train')
        self.val_writer = tf.summary.create_file_writer(logs_dir + '/validation')

        # checkpoint writer
        checkpoint_dir = logs_dir + '/checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              autoencoder=self.autoencoder)
        # restore from checkpoint if exists
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        if reset_optimizer:
            K.set_value(self.optimizer.iterations, 0)

    def train(self,
              loss_weights,
              norm_ord,
              do_augmentations,
              validation_freq,
              validation_steps,
              save_checkpoint_every_n_steps):
        """
        Trainer function for the AE model, all function parameters explanations are detailed in the
        ae_train_main.py flags.
        The function flow is as follows:
        Training loop:
            each forward pass looks like this:
                - encode img with E(x) -> get latent variable z
                - decode latent variable z with G(z) -> get reconstructed img x_hat
                - feed x, x_hat to discriminator
            compute losses:
                - reconstruction loss L1(x-x_hat)
                - dssim loss (x, x_hat)

            - take optimizer step
            - write to tensorboard / save checkpoint every n steps
        """

        progres_bar = Progbar(self.training_steps)
        for step in range(self.training_steps):
            progres_bar.update(step)

            img_batch, label_batch = next(self.train_generator)
            img_batch = tf.cast(img_batch, tf.float32)

            if do_augmentations:
                img_batch = augment(img_batch)

            with tf.GradientTape() as tape:
                # noise image (for denoising encoding)
                noisy = self.add_random_noise(img_batch)
                x_hat = self.autoencoder(noisy, training=True)

                # compute loss
                w_rec, w_dssim = loss_weights
                loss_rec = tf.reduce_mean(
                    tf.norm(img_batch - x_hat, ord=norm_ord, axis=(1, 2)))
                loss_dssim = tf.reduce_mean(
                    (1 - tf.image.ssim((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1)) / 2)
                psnr = tf.reduce_mean(
                    tf.image.psnr((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1))

                train_loss = (w_rec * loss_rec) + (w_dssim * loss_dssim)

                # compute gradients
                grad = tape.gradient(train_loss, self.autoencoder.trainable_variables)

                # apply gradients
                self.optimizer.apply_gradients(zip(grad, self.autoencoder.trainable_variables))

            # save checkpoint
            if step % save_checkpoint_every_n_steps == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # eval phase
            step = self.optimizer.iterations
            if step % validation_freq == 0:
                print(' \n Starting eval phase...')

                # write train summaries
                with self.train_writer.as_default():
                    tf.summary.scalar("loss_reconstruction", loss_rec, step=step)
                    tf.summary.scalar("loss_dssim", loss_dssim, step=step)
                    tf.summary.scalar("psnr", psnr, step=step)
                    tf.summary.scalar("learning_rate", self.lr_schedule(step), step=step)

                    # [-1, 1] -> [0, 255]
                    orig_display = tf.cast((noisy + 1) * 127.5, tf.uint8)
                    rec_display = tf.cast((x_hat + 1) * 127.5, tf.uint8)
                    concatenated_img = tf.concat([orig_display, rec_display], axis=2)
                    tf.summary.image('Noisy | Reconstructed', concatenated_img, step=step, max_outputs=8)

                self.eval(validation_steps, step)

    def eval(self, validation_steps, training_step):

        val_loss_rec = []
        val_loss_dssim = []
        val_psnr = []

        for _ in range(validation_steps):
            img_batch, label_batch = next(self.val_generator)
            img_batch = tf.cast(img_batch, tf.float32)

            # random invert
            img_batch = self.random_invert(img_batch)

            # autoencoder
            x_hat = self.autoencoder(img_batch, training=False)

            # compute loss
            loss_rec = tf.norm(img_batch - x_hat, ord=1, axis=(1, 2))
            loss_dssim = (1 - tf.image.ssim((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1)) / 2
            psnr = tf.image.psnr((img_batch + 1) / 2, (x_hat + 1) / 2, max_val=1)

            # aggregate
            val_loss_rec.extend(loss_rec)
            val_loss_dssim.extend(loss_dssim)
            val_psnr.extend(psnr)

        # compute mean loss over all validation steps
        val_loss_rec = tf.reduce_mean(
            tf.concat(loss_rec, axis=0))
        val_loss_dssim = tf.reduce_mean(
            tf.concat(loss_dssim, axis=0))
        val_psnr = tf.reduce_mean(val_psnr)

        with self.val_writer.as_default():
            tf.summary.scalar("loss_reconstruction", val_loss_rec, step=training_step)
            tf.summary.scalar("loss_dssim", val_loss_dssim, step=training_step)
            tf.summary.scalar("psnr", val_psnr, step=training_step)

            # [-1, 1] -> [0, 255]
            orig_display = tf.cast((img_batch + 1) * 127.5, tf.uint8)
            rec_display = tf.cast((x_hat + 1) * 127.5, tf.uint8)
            concatenated_img = tf.concat([orig_display, rec_display], axis=2)
            tf.summary.image('Original | Reconstructed', concatenated_img, step=training_step, max_outputs=16)

        print('Resuming training...')

    def add_random_noise(self, img_batch, prob_to_noise=0.66):
        tensors = tf.unstack(img_batch, axis=0)
        for i in range(len(tensors)):
            prob = tf.random.uniform(shape=[2, ], minval=0.0, maxval=1.0, dtype=tf.float32)
            if prob_to_noise > prob[0]:
                if prob[1] > 0.5:
                    tensors[i] = self.gaussian_noise(tensors[i])
                else:
                    tensors[i] = self.salt_and_pepper(tensors[i])

        img_batch = tf.stack(tensors)
        return img_batch

    @staticmethod
    def gaussian_noise(img_batch, stddev=0.5):
        noise = tf.random.normal(shape=img_batch.shape, mean=0.0, stddev=stddev)
        img_batch += noise
        img_batch = tf.clip_by_value(img_batch, clip_value_min=-1.0, clip_value_max=1.0)
        return img_batch

    @staticmethod
    def salt_and_pepper(img_batch, p=0.75):
        gaussian_noise = tf.random.normal(shape=img_batch.shape, mean=0.0, stddev=0.5)
        salt_mask = tf.cast(gaussian_noise > p, tf.float32) * 2
        pepper_mask = tf.cast(gaussian_noise <= -p, tf.float32) * -2
        img_batch = img_batch + salt_mask
        img_batch = img_batch + pepper_mask
        img_batch = tf.clip_by_value(img_batch, clip_value_min=-1.0, clip_value_max=1.0)
        return img_batch

    @staticmethod
    def random_invert(img_batch):
        tensors = tf.unstack(img_batch, axis=0)
        for i in range(len(tensors)):
            # random invert 50/50
            if tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5:
                tensors[i] = -tensors[i]
        img_batch = tf.stack(tensors)
        return img_batch
