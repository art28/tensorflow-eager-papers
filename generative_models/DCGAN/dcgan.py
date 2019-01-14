import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style

class Generator(tf.keras.Model):
    """ Generator Module for GAN
    Args:
        noise_dim: dimension of noise z. basically 100 is used
        output_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 output_dim=(28,28,1)):
        super(Generator, self).__init__()

        self.output_dim = output_dim

        self.dense_G_1 = tf.layers.Dense(49*256, activation=tf.nn.relu)
        self.bn1 = tf.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        self.deconv1 = tf.layers.Conv2DTranspose(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.bn2 = tf.layers.BatchNormalization()
        self.deconv2 = tf.layers.Conv2DTranspose(64, (3, 3), (2, 2), padding="same", activation=tf.nn.relu)
        self.bn3 = tf.layers.BatchNormalization()
        self.deconv3 = tf.layers.Conv2DTranspose(32,(3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.bn4 = tf.layers.BatchNormalization()
        self.deconv4 = tf.layers.Conv2DTranspose(1, (3, 3), (2, 2), padding="same", activation=tf.nn.sigmoid)

    def generate(self, Z, training):
        """
        Args:
            Z : input noise
        Return:
            gen : generated image
        """
        gen = self.dense_G_1(Z)

        gen = self.bn1(gen, training)
        gen = self.reshape(gen)

        gen = self.deconv1(gen)
        gen = self.bn2(gen, training)

        gen = self.deconv2(gen)
        gen = self.bn3(gen, training)

        gen = self.deconv3(gen)
        gen = self.bn4(gen, training)

        gen = self.deconv4(gen)

        return gen

    def call(self, Z , training):
        return self.generate(Z, training)


class Discriminator(tf.keras.Model):
    """ Discriminator Module for GAN
        get 28*28 image
        returns 1 for real image, 0 for zero image
    Args:
        input_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 input_dim=(28,28,1)):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim

        self.conv1 = tf.layers.Conv2D(32, (3, 3), (2, 2), padding="valid", activation=tf.nn.leaky_relu)
        self.batch1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.leaky_relu)
        self.batch2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(64,(3, 3), (2, 2), padding="valid", activation=tf.nn.leaky_relu)
        self.batch3 = tf.layers.BatchNormalization()
        self.conv4 = tf.layers.Conv2D(32,(3, 3), (1, 1), padding="same", activation=tf.nn.leaky_relu)
        self.batch4 = tf.layers.BatchNormalization()
        self.conv5 = tf.layers.Conv2D(16, (3, 3), (2, 2), padding="valid", activation=tf.nn.sigmoid)
        self.batch5 = tf.layers.BatchNormalization()
        self.conv6 = tf.layers.Conv2D(1, (2, 2), (1, 1), padding="valid")

        self.flatten = tf.layers.Flatten()

    def discriminate(self, X, training):
        """
        Args:
            X : input image
        Return:
            x: sigmoid logit[0, 1]
        """
        x = self.conv1(X)

        x = self.batch1(x, training)
        x = self.conv2(x)

        x = self.batch2(x, training)
        x = self.conv3(x)

        x = self.batch3(x, training)
        x = self.conv4(x)

        x = self.batch4(x, training)
        x = self.conv5(x)

        x = self.batch5(x, training)
        x = self.conv6(x)

        x = self.flatten(x)

        return x

    def call(self, X, training):
        return self.discriminate(X, training)


class DCGAN(tf.keras.Model):
    """ Generative Adversarial Network model for mnist dataset.
    Args:
        noise_dim: dimension of noise z. Basically 100.
        output_dim: dimension of output image. 28*28 in MNIST.
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 noise_dim=100,
                 output_dim=(28,28,1),
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(DCGAN, self).__init__()

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
        fake = self.generator(Z, training)
        logits_fake = self.discriminator(fake, training)
        return fake, logits_fake

    def loss_G(self, Z, training):
        """calculate loss of generator
        Args:
            Z : noise vector
            training: option for batch-normalization
        """
        fake = self.generator(Z, training)
        logits = self.discriminator(fake, training)

        loss_val = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits)

        return loss_val

    def loss_D(self, Z, real, training):
        """calculate loss of discriminator
        Args:
            Z : noise vector
            real : real image
            training: option for batch-normalization
        """
        fake = self.generator(Z, training)
        logits_fake = self.discriminator(fake, training)
        logits_real = self.discriminator(real, training)

        loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)

        loss_val = 0.5 * tf.add(loss_fake, loss_real)

        return loss_val

    def grad_G(self, Z, training):
        """calculate gradient of the batch for generator
        Args:
            Z : noise vector
            training: option for batch-normalization
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_G(Z, training)
        return tape.gradient(loss_val, self.generator.variables), loss_val

    def grad_D(self, Z, real, training):
        """calculate gradient of the batch for discriminator
        Args:
            Z: noise vector
            real: real image
            training: option for batch-normalization
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_D(Z, real, training)
        return tape.gradient(loss_val, self.discriminator.variables), loss_val

    def grad_both(self, Z, real, training):
        """calculate gradient of the batch for both generator and discriminator
        Args:
            Z: noise vector
            real: real image
            training: option for batch-normalization
        """
        with tfe.GradientTape(persistent=True) as tape:
            loss_G = self.loss_G(Z, training)
            loss_D = self.loss_D(Z, real, training)
        return tape.gradient(loss_G, self.generator.variables), tape.gradient(loss_D, self.discriminator.variables), loss_G, loss_D

    def fit(self, X_train, X_val, epochs=1, verbose=1, both_step=1, gen_step=0, batch_size=32, saving=False,
            tqdm_option=None):
        """train the GAN network
        Args:
            train_data: train dataset
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
                time.sleep(1)

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        dummy_input_G = tf.zeros((1, self.noise_dim))
        dummy_input_D = tf.zeros((1,)+ self.output_dim)

        dummy_img = self.generator(dummy_input_G, True)
        dummy_logit = self.discriminator(dummy_input_D, True)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)
