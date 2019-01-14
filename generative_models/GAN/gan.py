import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style


class Generator(tf.keras.Model):
    """ Generator Module for GAN
    Args:
        noise_dim: dimension of noise z. basically 10 is used
        output_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 output_dim=28 * 28):
        super(Generator, self).__init__()

        self.output_dim = output_dim

        self.dense_G_1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dense_G_2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.fake = tf.layers.Dense(self.output_dim, activation=tf.nn.tanh)

    def generate(self, Z):
        """
        Args:
            Z : input noise
        Return:
            gen : generated image
        """

        gen = self.dense_G_1(Z)
        gen = self.dense_G_2(gen)
        gen = self.fake(gen)

        return gen

    def call(self, Z):
        return self.generate(Z)


class Discriminator(tf.keras.Model):
    """ Discriminator Module for GAN
        get 28*28 image
        returns 1 for real image, 0 for zero image
    Args:
        input_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 input_dim=28 * 28):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim

        self.dense_D_1 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.dropout1 = tf.layers.Dropout(0.7)
        self.dense_D_2 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = tf.layers.Dropout(0.7)
        self.dense_D_3 = tf.layers.Dense(32, activation=tf.nn.relu)
        self.discrimination = tf.layers.Dense(1, activation=tf.nn.sigmoid)

    def discriminate(self, X, training):
        """
        Args:
            X : input image
        Return:
            x: sigmoid logit[0, 1]
        """
        x = self.dense_D_1(X)
        x = self.dense_D_2(x)
        if training:
            x = self.dropout1(x)
        x = self.dense_D_3(x)
        if training:
            x = self.dropout2(x)
        x = self.discrimination(x)

        return x

    def call(self, X, training):
        return self.discriminate(X, training)


class GAN(tf.keras.Model):
    """ Generative Adversarial Network model for mnist dataset.
    Args:
        noise_dim: dimension of noise z. Basically 10.
        output_dim: dimension of output image. 28*28 in MNIST.
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 noise_dim=10,
                 output_dim=28 * 28,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(GAN, self).__init__()

        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        self.generator = Generator(self.output_dim)
        self.discriminator = Discriminator(self.output_dim)

        # optimizer
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.global_step = 0

    def call(self, Z, training):
        fake = self.generator(Z)
        logits_fake = self.discriminator(fake, training)

        return fake, logits_fake

    def loss_G(self, Z, training):
        """calculate loss of generator
        Args:
            Z : noise vector
        """
        fake = self.generator(Z)
        logits = self.discriminator(fake, training)

        loss_val = -1. * tf.log(logits)
        # loss_val = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits)

        return loss_val

    def loss_D(self, Z, real, training):
        """calculate loss of discriminator
        Args:
            Z : noise vector
            real : real image
        """
        fake = self.generator(Z)
        logits_fake = self.discriminator(fake, training)
        logits_real = self.discriminator(real, training)

        loss_fake = -1. * tf.reduce_mean(tf.log(1 - logits_fake + 1e-8))
        loss_real = -1. * tf.reduce_mean(tf.log(logits_real + 1e-8))

        # loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        # loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)

        loss_val = 0.5 * tf.add(loss_fake, loss_real)

        return loss_val

    def grad_G(self, Z, training):
        """calculate gradient of the batch for generator
        Args:.
            Z : noise vector
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_G(Z, training)
        return tape.gradient(loss_val, self.generator.variables), loss_val

    def grad_D(self, Z, real, training):
        """calculate gradient of the batch for discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_D(Z, real, training)
        return tape.gradient(loss_val, self.discriminator.variables), loss_val

    def grad_both(self, Z, real, training):
        """calculate gradient of the batch for both generator and discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape(persistent=True) as tape:
            loss_G = self.loss_G(Z, training)
            loss_D = self.loss_D(Z, real, training)
        return tape.gradient(loss_G, self.generator.variables), tape.gradient(loss_D,
                                                                              self.discriminator.variables), loss_G, loss_D

    def fit(self, X_train, X_val, epochs=1, verbose=1, both_step=1, gen_step=0, batch_size=32, saving=False,
            tqdm_option=None):
        """train the GAN network
        Args:
            X_train : train dataset input
            X_val : validation dataset input
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            gen_step & both_step : step distribution, on both_step, both discriminator and generator learns, and on gen_step, only generator learns
            batch_size : training batch size
            saving: whether to save checkpoint or not
        """

        def tqdm_wrapper(*args, **kwargs):
            if tqdm_option == "normal":
                return tqdm(*args, **kwargs)
            elif tqdm_option == "notebook":
                return tqdm_notebook(*args, **kwargs)
            else:
                return args[0]

        with tf.device(self.device_name):
            ds_train = tf.data.Dataset.from_tensor_slices((X_train,)).shuffle(99999999).batch(batch_size)
            batchlen_train = (len(X_train) - 1) // batch_size + 1
            ds_val = tf.data.Dataset.from_tensor_slices((X_val,)).shuffle(99999999).batch(batch_size)
            batchlen_val = (len(X_val) - 1) // batch_size + 1

            for i in range(epochs):
                self.global_step += 1
                epoch_loss_G = 0.0
                epoch_loss_D = 0.0

                for (X,) in tqdm_wrapper(ds_train, total=batchlen_train, desc="TRAIN%-2s" % self.global_step):
                    Z = tf.random_normal((X.shape[0], self.noise_dim))

                    if ((self.global_step-1) % (gen_step + both_step)) < both_step:
                        grads_G, grads_D, batch_loss_G, batch_loss_D = self.grad_both(Z, X, True)
                        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.variables))
                        self.optimizer_D.apply_gradients(zip(grads_D, self.discriminator.variables))
                        epoch_loss_G += tf.reduce_mean(batch_loss_G)
                        epoch_loss_D += tf.reduce_mean(batch_loss_D)
                    else:
                        grads_G, batch_loss_G = self.grad_G(Z, True)
                        batch_loss_D = self.loss_D(Z, X, False)
                        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.variables))
                        epoch_loss_G += tf.reduce_mean(batch_loss_G)
                        epoch_loss_D += tf.reduce_mean(batch_loss_D)

                epoch_loss_G_val = 0.0
                epoch_loss_D_val = 0.0
                for (X,) in tqdm_wrapper(ds_val, total=batchlen_val, desc="VAL  %-2s" % self.global_step):
                    Z = tf.random_normal((X.shape[0], self.noise_dim))
                    batch_loss_G, batch_loss_D = self.loss_G(Z, False), self.loss_D(Z, X, False)
                    epoch_loss_G_val += tf.reduce_mean(batch_loss_G)
                    epoch_loss_D_val += tf.reduce_mean(batch_loss_D)

                if i == 0 or ((i + 1) % verbose == 0):
                    if ((self.global_step-1) % (gen_step + both_step)) < both_step:
                        gen_step_notice = ""
                    else:
                        gen_step_notice = "-- Generator only step"
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d] TRAIN %s" % ((i + 1), self.global_step, gen_step_notice))
                    print("TRAIN loss   : %.4f" % ((epoch_loss_G + epoch_loss_D) / batchlen_train))
                    print("GEN   loss   : %.4f" % (epoch_loss_G / batchlen_train))
                    print("DIS   loss   : %.4f" % (epoch_loss_D / batchlen_train))
                    print("=" * 25 + Style.RESET_ALL)
                    print(Fore.BLUE + "=" * 25)
                    print("[EPOCH %d / STEP %d] VAL %s" % ((i + 1), self.global_step, gen_step_notice))
                    print("TRAIN loss   : %.4f" % ((epoch_loss_G_val + epoch_loss_D_val) / batchlen_val))
                    print("GEN   loss   : %.4f" % (epoch_loss_G_val / batchlen_val))
                    print("DIS   loss   : %.4f" % (epoch_loss_D_val / batchlen_val))
                    print("=" * 25 + Style.RESET_ALL)

                    if saving:
                        import os
                        if not os.path.exists(self.checkpoint_directory):
                            os.makedirs(self.checkpoint_directory)
                        self.save()

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        dummy_input_G = tf.zeros((1, self.noise_dim))
        dummy_input_D = tf.zeros((1, self.output_dim))

        dummy_img = self.generator(dummy_input_G)
        dummy_logit = self.discriminator(dummy_input_D, True)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)
