import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style

class VAE(tf.keras.Model):
    """ Variational Autoencoder model for mnist dataset.
    Args:
        input_dim: dimension of input. (28 * 28) for mnist(height * width)
        z_dim : dimension of z, which is compressed feature vector of input
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """
    def __init__(self, input_dim = 28*28,
                 z_dim = 10,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # Encoder layers
        self.encode_dense1 = tf.layers.Dense(512, activation=tf.nn.elu)
        self.encode_dense2 = tf.layers.Dense(384, activation=tf.nn.elu)
        self.encode_dense3 = tf.layers.Dense(256, activation=tf.nn.elu)
        self.encode_mu = tf.layers.Dense(z_dim)
        self.encode_logsigma = tf.layers.Dense(z_dim)

        # Decoder layers
        self.decode_dense1 = tf.layers.Dense(256, activation=tf.nn.elu)
        self.decode_dense2 = tf.layers.Dense(384, activation=tf.nn.elu)
        self.decode_dense3 = tf.layers.Dense(512, activation=tf.nn.elu)
        self.decode_out_layer = tf.layers.Dense(self.input_dim)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.global_step = 0

    def encoding(self, X):
        """encoding input data to normal distribution
        Args:
            X : input tensor
        Returns:
            mu : mean of distribution
            logsigma : log value of variation of distribution
        """
        z = self.encode_dense1(X)
        z = self.encode_dense2(z)
        z = self.encode_dense3(z)
        mu = self.encode_mu(z)
        logsigma = self.encode_logsigma(z)

        return mu, logsigma

    def sampling_z(self, z_mu, z_logsigma):
        """sampling z using mu and logsigma, using reparameterization trick
        Args:
            z_mu : mean of distribution
            z_logsigma : log value of variation of distribution
        Return:
            z value
        """
        epsilon = tf.random_normal(shape=tf.shape(z_mu), dtype=tf.float32)
        return z_mu + tf.exp(z_logsigma*0.5) * epsilon

    def decoding(self, Z):
        """image generation using z value
        Args:
            Z : z value, which is compressed feature part of the data
        Returns:
            x_decode : generated image
            sigmoid(x_decode) : generated image + sigmoid activation
        """
        x_decode = self.decode_dense1(Z)
        x_decode = self.decode_dense2(x_decode)
        x_decode = self.decode_dense3(x_decode)
        x_decode = self.decode_out_layer(x_decode)

        return x_decode

    def call(self, X):
        mu, logsigma = self.encoding(X)
        Z = self.sampling_z(mu, logsigma)
        X_decode = self.decoding(Z)
        return X_decode

    def loss(self, X):
        """calculate loss of VAE model
        Args:
            X : original image batch
        """
        mu, logsigma = self.encoding(X)
        Z = self.sampling_z(mu, logsigma)
        X_decode = self.decoding(Z)


        # what sigmoid_corss_entropy do
        # 1. cross entropy of [sigmoid(logits) & labels]
        # 2. mean of dimensions(input_dim)
        # 3. mean of batches
        # we only need 1 & 3 so revert 2 by multiplying input_dim
        reconstruction_loss = self.input_dim * tf.losses.sigmoid_cross_entropy(logits=X_decode, multi_class_labels=X)

        kl_div = - 0.5 * tf.reduce_sum(1. + logsigma - tf.square(mu) - tf.exp(logsigma), axis=1)

        total_loss = reconstruction_loss + kl_div

        return total_loss, reconstruction_loss, kl_div

    def grad(self, X):
        """calculate gradient of the batch
        Args:
            X : input tensor
        """
        with tfe.GradientTape() as tape:
            loss_val, recon_loss, kl_loss = self.loss(X)
        return tape.gradient(loss_val, self.variables), loss_val, recon_loss, kl_loss

    def fit(self, X_train, X_val, epochs=1, verbose=1, batch_size=32, saving=False, tqdm_option=None):
        """train the network
        Args:
            X_train : train dataset input
            X_val : validation dataset input
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            batch_size : training batch size
            saving: whether to save checkpoint or not
            tqdm_option: tqdm logger option. default is none. use "normal" for tqdm, "notebook" for tqdm_notebook

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
            batchlen_train = (len(X_train)-1)//batch_size + 1
            ds_val = tf.data.Dataset.from_tensor_slices((X_val,)).batch(batch_size)
            batchlen_val = (len(X_val)-1)//batch_size + 1

            for i in range(epochs):
                epoch_loss = 0.
                epoch_reconstruction_loss = 0.
                epoch_kl_loss = 0.
                self.global_step += 1

                for (X,) in tqdm_wrapper(ds_train, total=batchlen_train, desc="TRAIN%-2s" % self.global_step):
                    grads, batch_loss, recon_loss, kl_loss = self.grad(X)
                    mean_loss = tf.reduce_mean(batch_loss)
                    mean_loss_recon = tf.reduce_mean(recon_loss)
                    mean_loss_kl = tf.reduce_mean(kl_loss)
                    epoch_loss += mean_loss
                    epoch_reconstruction_loss += mean_loss_recon
                    epoch_kl_loss += mean_loss_kl
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                epoch_loss_val = 0.
                epoch_reconstruction_loss_val = 0.
                epoch_kl_loss_val = 0.

                for (X,) in tqdm_wrapper(ds_val, total=batchlen_val, desc="VAL  %-2s" % self.global_step):
                    batch_loss, recon_loss, kl_loss = self.loss(X)
                    mean_loss = tf.reduce_mean(batch_loss)
                    mean_loss_recon = tf.reduce_mean(recon_loss)
                    mean_loss_kl = tf.reduce_mean(kl_loss)
                    epoch_loss_val += mean_loss
                    epoch_reconstruction_loss_val += mean_loss_recon
                    epoch_kl_loss_val += mean_loss_kl

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d] TRAIN" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss/batchlen_train))
                    print("RECON loss   : %.4f" % (epoch_reconstruction_loss/batchlen_train))
                    print("KL    loss   : %.4f" % (epoch_kl_loss/batchlen_train))
                    print("=" * 25 + Style.RESET_ALL)
                    print(Fore.BLUE + "=" * 25)
                    print("[EPOCH %d / STEP %d] VAL  " % ((i + 1), self.global_step))
                    print("VAL   loss   : %.4f" % (epoch_loss_val/batchlen_val))
                    print("RECON loss   : %.4f" % (epoch_reconstruction_loss_val/batchlen_val))
                    print("KL    loss   : %.4f" % (epoch_kl_loss_val/batchlen_val))
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
        dummy_input = tf.zeros((1, self.input_dim))
        dummy_mu, dummy_sigma = self.encoding(dummy_input)
        dummy_z = self.sampling_z(dummy_mu, dummy_sigma)
        dummy_ret = self.decoding(dummy_z)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)