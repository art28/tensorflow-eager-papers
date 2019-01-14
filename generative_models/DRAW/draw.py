import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style


class DRAW(tf.keras.Model):
    """ DRAW Variational Autoencoder model for mnist dataset.
    Args:
        A : length of image's x-axis
        B : length of image's y-axis
        T : length of [fixed] attention sequence for LSTM
        read_N : reading attention filter's size
        write_N : writing attention filter's size
        z_dim : dimension of z, which is compressed feature vector of input
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 A=28,
                 B=28,
                 z_dim=10,
                 T=10,
                 read_N=5,
                 write_N=5,
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 use_cudnn=False,
                 device_name="cpu:0"):
        super(DRAW, self).__init__()

        self.A = A
        self.B = B
        self.input_dim = self.A * self.B
        self.z_dim = z_dim
        self.T = T
        self.read_N = read_N
        self.write_N = write_N

        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.use_cudnn = use_cudnn
        self.device_name = device_name

        # Reading attention network
        self.read_attention_params = tf.layers.Dense(5, name="dense_read_attention_params")

        # Encoder layers
        if self.use_cudnn:
            self.lstm_encode = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(256)
        else:
            self.lstm_encode = tf.nn.rnn_cell.LSTMCell(256)

        self.encode_mu = tf.layers.Dense(z_dim, name="dense_encode_mu")
        self.encode_logsigma = tf.layers.Dense(z_dim, name="dense_encode_logsigma")

        # Decoder layers
        if self.use_cudnn:
            self.lstm_decode = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(256)
        else:
            self.lstm_decode = tf.nn.rnn_cell.LSTMCell(256)
        # Writing attention network
        self.write_attention_params = tf.layers.Dense(5, name="dense_write_attention_params")
        self.write_attention_dense = tf.layers.Dense(self.write_N * self.write_N, name="dense_write_attention_dense")

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.global_step = 0

    def call(self, X):
        batch_size = X.shape[0]
        x = X
        canvas = [0] * self.T
        encode_state = self.lstm_encode.zero_state(batch_size, tf.float32)
        decode_state = self.lstm_decode.zero_state(batch_size, tf.float32)
        prev_out_decode = tf.zeros((batch_size, 256))  # size of decode lstm cell

        for t in range(self.T):
            c_prev = tf.zeros((batch_size, self.A * self.B), tf.float32) if t == 0 else canvas[t - 1]
            x_hat = x - tf.sigmoid(c_prev)
            attention_read = self.read(x, x_hat, prev_out_decode)
            encode_state, z_mu, z_logsigma = self.encoding(encode_state, attention_read)
            z = self.sampling_z(z_mu, z_logsigma)
            out_decode, decode_state = self.decoding(decode_state, z)
            canvas[t] = c_prev + self.write(out_decode)
            prev_out_decode = out_decode

        return canvas


    def read(self, X, X_hat, decode_out):
        """process reading, using original data(X) and error term(X_hat)
        Args:
            X : input tensor of original data
            X_hat : error term, difference between former canvas
            decode_out : former output(hidden) from decoding layer
        Returns:
            reading attention of X and X_hat, with concatenation.[2*read_N * read_N]
        """
        # parameter inference
        parameters = self.read_attention_params(decode_out)
        gx_, gy_, logsigma_sq, logdelta, loggamma = tf.split(parameters, 5, axis=1)
        gx = ((self.A + 1) / 2) * (gx_ + 1)
        gy = ((self.B + 1) / 2) * (gy_ + 1)
        delta = ((max(self.A, self.B) - 1) / (self.read_N - 1)) * tf.exp(logdelta)
        sigma_sq = tf.exp(logsigma_sq)
        gamma = tf.exp(loggamma)

        # filterbank
        grid = tf.reshape(tf.cast(tf.range(self.read_N), tf.float32), (1, -1))
        mu_x = gx + (grid - self.read_N / 2 - 0.5) * delta
        mu_y = gy + (grid - self.read_N / 2 - 0.5) * delta
        a = tf.reshape(tf.cast(tf.range(self.A), tf.float32), (1, 1, -1))
        b = tf.reshape(tf.cast(tf.range(self.B), tf.float32), (1, 1, -1))
        mu_x = tf.reshape(mu_x, [-1, self.read_N, 1])
        mu_y = tf.reshape(mu_y, [-1, self.read_N, 1])
        sigma_sq = tf.reshape(sigma_sq, [-1, 1, 1])

        Fx_unnormalized = tf.exp(-tf.square(a - mu_x) / (2 * sigma_sq))
        Fy_unnormalized = tf.exp(-tf.square(b - mu_y) / (2 * sigma_sq))
        Zx = tf.maximum(tf.reduce_sum(Fx_unnormalized, 2, keep_dims=True), 1e-8)
        Zy = tf.maximum(tf.reduce_sum(Fy_unnormalized, 2, keep_dims=True), 1e-8)
        Fx = Fx_unnormalized / Zx
        Fy = Fy_unnormalized / Zy

        # attention(original)
        Fx_T = tf.transpose(Fx, perm=[0, 2, 1])  # transpose but fix batch dimension. (Batch_size * (A * read_N))
        x = tf.reshape(X, (-1, self.B, self.A))  # image batch (Batch_size * (B * read_A))
        # and Fy is (Batch_size * (read_N * B))
        attention = tf.matmul(Fy, tf.matmul(x, Fx_T)) * tf.reshape(gamma, (X.shape[0], -1, 1))

        # attention(error)
        x_hat = tf.reshape(X_hat, (-1, self.B, self.A))  # image batch (Batch_size * (B * read_A))
        attention_hat = tf.matmul(Fy, tf.matmul(x_hat, Fx_T)) * tf.reshape(gamma, (X.shape[0], -1, 1))

        return tf.concat([tf.reshape(attention, (-1, self.read_N * self.read_N)),
                          tf.reshape(attention_hat, (-1, self.read_N * self.read_N))], axis=1)

    def encoding(self, encode_state, attention):
        """encoding input data to normal distribution
        Args:
            encode_state : former state of encoding LSTM cell
            attention : computed reading attention from read operation
        Returns:

            z_mu : mean of latent variable distribution
            z_logsigma : log value of standard deviation,  distribution
        """
        out_encode, encode_state = self.lstm_encode(attention, encode_state)
        z_mu = self.encode_mu(out_encode)
        z_logsigma = self.encode_logsigma(out_encode)

        return encode_state, z_mu, z_logsigma

    def sampling_z(self, z_mu, z_logsigma):
        """sampling z using mu and logsigma, using reparameterization trick
        Args:
            z_mu : mean of distribution
            z_logsigma : log value of sigma of latent variable distribution
        Return:
            z value
        """
        epsilon = tf.random_normal(shape=tf.shape(z_mu), dtype=tf.float32)
        return z_mu + tf.exp(z_logsigma * 0.5) * epsilon

    def decoding(self, decode_state, z):
        """decoding latent variable to hidden variable for writing and reading attention
        Args:
            decode_state : former state of decoding LSTM cell
            z : computed value of latent variable
        Returns:
            out_decode : hidden variable to use in attention calculation
            decode_state : state variable for decoding LSTM cell
        """
        out_decode, decode_state = self.lstm_decode(z, decode_state)

        return out_decode, decode_state

    def write(self, decode_out):
        """process writing, using hidden variable from decoding LSTM cell
        Args:
            decode_out : former output(hidden) from decoding layer
        Returns:
            writing attention, resized to image size(B * A)
        """
        parameters = self.write_attention_params(decode_out)
        gx_, gy_, logsigma_sq, logdelta, loggamma = tf.split(parameters, 5, axis=1)
        gx = ((self.A + 1) / 2) * (gx_ + 1)
        gy = ((self.B + 1) / 2) * (gy_ + 1)
        delta = ((max(self.A, self.B) - 1) / (self.write_N - 1)) * tf.exp(logdelta)
        sigma_sq = tf.exp(logsigma_sq)
        gamma = tf.exp(loggamma)

        # filterbank
        grid = tf.reshape(tf.cast(tf.range(self.write_N), tf.float32), (1, -1))
        mu_x = gx + (grid - self.write_N / 2 - 0.5) * delta
        mu_y = gy + (grid - self.write_N / 2 - 0.5) * delta
        a = tf.reshape(tf.cast(tf.range(self.A), tf.float32), (1, 1, -1))
        b = tf.reshape(tf.cast(tf.range(self.B), tf.float32), (1, 1, -1))
        mu_x = tf.reshape(mu_x, [-1, self.write_N, 1])
        mu_y = tf.reshape(mu_y, [-1, self.write_N, 1])
        sigma_sq = tf.reshape(sigma_sq, [-1, 1, 1])
        Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma_sq))
        Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma_sq))
        Zx = tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 1e-8)
        Zy = tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 1e-8)
        Fx = Fx / Zx
        Fy = Fy / Zy

        writing = self.write_attention_dense(decode_out)

        Fy_T = tf.transpose(Fy, perm=[0, 2, 1])  # transpose but fix batch dimension. (Batch_size * (B * write_N))
        x = tf.reshape(writing, (-1, self.write_N, self.write_N))  # writing batch (Batch_size * (write_N * write_N))
        # and Fx is (Batch_size * (write_N * A))
        attention = tf.matmul(Fy_T, tf.matmul(x, Fx))
        attention = tf.reshape(attention, (-1, self.A * self.B)) * tf.reshape(1.0 / gamma, (-1, 1))

        return attention

    def extract(self, X):
        """ testing method, to extract z from image batch
        Args:
            X : input image
        Returns:
            extracted latent variable z
        """
        batch_size = X.shape[0]
        x = tf.reshape(X, (-1, self.A * self.B))
        canvas = [0] * self.T
        encode_state = self.lstm_encode.zero_state(batch_size, tf.float32)
        decode_state = self.lstm_decode.zero_state(batch_size, tf.float32)
        prev_out_decode = tf.zeros((batch_size, 256))  # size of decode lstm cell

        for t in range(self.T):
            c_prev = tf.zeros((batch_size, self.A * self.B), tf.float32) if t == 0 else canvas[t - 1]
            x_hat = x - tf.sigmoid(c_prev)
            attention_read = self.read(x, x_hat, prev_out_decode)
            encode_state, z_mu, z_logsigma = self.encoding(encode_state, attention_read)
            z = self.sampling_z(z_mu, z_logsigma)
            out_decode, decode_state = self.decoding(decode_state, z)
            canvas[t] = c_prev + self.write(out_decode)
            prev_out_decode = out_decode

        return z

    def draw(self, z):
        """ testing method, generate image from latent variable z
        Args:
            z : latent variable
        Returns:
            generated canvas images
        """
        batch_size = z.shape[0]
        canvas = [0] * self.T
        decode_state = self.lstm_decode.zero_state(batch_size, tf.float32)

        for t in range(self.T):
            c_prev = tf.zeros((batch_size, self.A * self.B)) if t == 0 else canvas[t - 1]
            out_decode, decode_state = self.decoding(decode_state, z)
            canvas[t] = c_prev + self.write(out_decode)

        return canvas

    def loss(self, X):
        """calculate loss of DRAW model
        Args:
            X : original image batch
        """
        batch_size = X.shape[0]
        x = X
        canvas = [0] * self.T
        encode_state = self.lstm_encode.zero_state(batch_size, tf.float32)
        decode_state = self.lstm_decode.zero_state(batch_size, tf.float32)
        prev_out_decode = tf.zeros((batch_size, 256))  # size of decode lstm cell

        kl_divs = []
        for t in range(self.T):
            c_prev = tf.zeros((batch_size, self.A * self.B), tf.float32) if t == 0 else canvas[t - 1]
            x_hat = x - tf.sigmoid(c_prev)
            attention_read = self.read(x, x_hat, prev_out_decode)
            encode_state, z_mu, z_logsigma = self.encoding(encode_state, attention_read)
            z = self.sampling_z(z_mu, z_logsigma)
            out_decode, decode_state = self.decoding(decode_state, z)
            canvas[t] = c_prev + self.write(out_decode)
            prev_out_decode = out_decode

            kl_divs.append(- 0.5 * tf.reduce_sum(1. + z_logsigma - tf.square(z_mu) - tf.exp(z_logsigma), axis=1))

        kl_div = tf.add_n(kl_divs)
        kl_div = tf.reduce_mean(kl_div)

        X_reconstructed = canvas[-1]

        reconstruction_loss = self.input_dim * tf.losses.sigmoid_cross_entropy(logits=X_reconstructed,
                                                                               multi_class_labels=X)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        total_loss = reconstruction_loss + kl_div

        return total_loss, reconstruction_loss, kl_div

    def grad(self, X):
        """calculate gradient of the batch
        Args:
            X : input tensor
        """
        with tfe.GradientTape() as tape:
            loss_val, loss_recon, loss_kl = self.loss(X)
        return tape.gradient(loss_val, self.variables), loss_val, loss_recon, loss_kl

    def fit(self, X_train, X_val, epochs=1, verbose=1, batch_size=32, saving=False, tqdm_option=None):
        """train the network
        Args:
            X_train : train dataset input
            X_val : validation dataset input
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
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

                epoch_loss = 0.0
                epoch_reconstruction_loss = 0.
                epoch_KL_loss = 0.
                for (X,) in tqdm_wrapper(ds_train, total=batchlen_train, desc="TRAIN %3s" % self.global_step):
                    grads, batch_loss, batch_loss_recon, batch_loss_kl = self.grad(X)
                    mean_loss = tf.reduce_mean(batch_loss)
                    mean_loss_recon = tf.reduce_mean(batch_loss_recon)
                    mean_loss_kl = tf.reduce_mean(batch_loss_kl)
                    epoch_loss += mean_loss
                    epoch_reconstruction_loss += mean_loss_recon
                    epoch_KL_loss += mean_loss_kl

                    # clipping gradient(you can un-comment it if you want)
                    # for i, g in enumerate(grads):
                    #     if g is not None:
                    #         grads[i] = tf.clip_by_norm(g, 5)

                    self.optimizer.apply_gradients(zip(grads, self.variables))

                epoch_loss_val = 0.0
                epoch_reconstruction_loss_val = 0.
                epoch_KL_loss_val = 0.
                for (X,) in tqdm_wrapper(ds_val, total=batchlen_val, desc="VAL   %3s" % self.global_step):
                    batch_loss, batch_loss_recon, batch_loss_kl = self.loss(X)
                    mean_loss = tf.reduce_mean(batch_loss)
                    mean_loss_recon = tf.reduce_mean(batch_loss_recon)
                    mean_loss_kl = tf.reduce_mean(batch_loss_kl)
                    epoch_loss_val += mean_loss
                    epoch_reconstruction_loss_val += mean_loss_recon
                    epoch_KL_loss_val += mean_loss_kl

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss / batchlen_train))
                    print("RECON loss   : %.4f" % (epoch_reconstruction_loss / batchlen_train))
                    print("KL    loss   : %.4f" % (epoch_KL_loss / batchlen_train))
                    print("=" * 25 + Style.RESET_ALL)
                    print(Fore.BLUE + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss_val / batchlen_val))
                    print("RECON loss   : %.4f" % (epoch_reconstruction_loss_val / batchlen_val))
                    print("KL    loss   : %.4f" % (epoch_KL_loss_val / batchlen_val))
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
        self.loss(dummy_input)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)
