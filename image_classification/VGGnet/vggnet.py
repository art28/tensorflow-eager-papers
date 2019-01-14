import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style


class VGGnet(tf.keras.Model):
    """ VGGnet model for CIFAR-10 dataset.
    Args:
        input_dim: dimension of input. (32, 32, 3) for CIFAR-10.(height - width - channel)
        out_dim: dimension of output. 10 class for CIFAR-10
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 input_dim=(32, 32, 3),
                 out_dim=10,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(VGGnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # layer declaration
        # use padding and restrict strides to (1,1)
        # because image is already too small to reduce more dimensions
        self.conv1a = tf.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                      activation=tf.nn.relu)
        self.conv1b = tf.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                      activation=tf.nn.relu)
        self.maxpool1 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.conv2a = tf.layers.Conv2D(16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv2b = tf.layers.Conv2D(16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool2 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.conv3a = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv3b = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv3c = tf.layers.Conv2D(32, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool3 = tf.layers.MaxPooling2D((3, 3), (2, 2))

        self.conv4a = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv4b = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv4c = tf.layers.Conv2D(64, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool4 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.conv5a = tf.layers.Conv2D(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv5b = tf.layers.Conv2D(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv5c = tf.layers.Conv2D(128, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool5 = tf.layers.MaxPooling2D((2, 2), (2, 2))

        self.flatten = tf.layers.Flatten()

        self.dense1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout1 = tf.layers.Dropout(0.5)

        self.dense2 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = tf.layers.Dropout(0.5)

        self.out_layer = tf.layers.Dense(self.out_dim)

        # optimizer
        # original paper use normal gradient descent algorithm,
        # but I highly recommend you to use momentum-based one for time-efficiency
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


        # global step
        self.global_step = 0


    def predict(self, X, training):
        """predicting output of the network
        Args:
            X : input tensor
            training : whether apply dropout or not
        """
        x = self.conv1a(X)
        x = self.conv1b(X)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.maxpool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.maxpool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.maxpool5(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)

        x = self.dense2(x)
        if training:
            x = self.dropout2(x)

        x = self.out_layer(x)

        return x

    def call(self, X, training):
        return self.predict(X, training)

    def loss(self, X, y, training):
        """calculate loss of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        prediction = self.predict(X, training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
        return loss_value, prediction

    def grad(self, X, y, trainig):
        """calculate gradient of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        with tfe.GradientTape() as tape:
            loss_value, _ = self.loss(X, y, trainig)
        return tape.gradient(loss_value, self.variables), loss_value


    def fit(self, X_train, y_train, X_val, y_val, epochs=1, verbose=1, batch_size=32, saving=False, tqdm_option=None):
        """train the network
        Args:
            X_train : train dataset input
            y_train : train dataset label
            X_val : validation dataset input
            y_val = validation dataset input
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

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(999999999).batch(batch_size)
        batchlen_train = (len(X_train) - 1) // batch_size + 1

        dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(999999999).batch(batch_size)
        batchlen_val = (len(X_val) - 1) // batch_size + 1

        with tf.device(self.device_name):
            for i in range(epochs):
                epoch_loss = 0.0
                self.global_step += 1
                for X, y in tqdm_wrapper(dataset_train, total=batchlen_train, desc="GLOBAL %s" % self.global_step):
                    grads, batch_loss = self.grad(X, y, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                epoch_loss_val = 0.0
                val_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_val, total=batchlen_val, desc="GLOBAL %s" % self.global_step):
                    batch_loss, pred = self.loss(X, y, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)
                    val_accuracy(tf.argmax(pred, axis=1), y)

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss / batchlen_train))
                    print("VAL   loss   : %.4f" % (epoch_loss_val / batchlen_val))
                    print("VAL   acc    : %.4f%%" % (val_accuracy.result().numpy() * 100))

                    if saving:
                        self.save()
                    print("=" * 25 + Style.RESET_ALL)
                time.sleep(1)

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        dummy_input = tf.constant(tf.zeros((1,) + self.input_dim))
        dummy_pred = self.call(dummy_input, True)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
