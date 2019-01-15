import time
import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

from tqdm import tqdm, tqdm_notebook
from colorama import Fore, Style


class CNN_character(tf.keras.Model):
    """ Character-level classifier for Movie Review dataset(single channel)
    Args:
        num_chars: uniwue chars in dataset.
        in_dim: dimension of input array, which is maximum word counts among texts.
        out_dim: softmax output dimension.
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 num_chars,
                 in_dim,
                 out_dim,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(CNN_character, self).__init__()
        self.num_chars = num_chars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        self.conv11 = tf.layers.Conv1D(filters=1024, kernel_size=7, padding="valid", name="conv11")
        self.conv12 = tf.layers.Conv1D(filters=1024, kernel_size=7, padding="valid", name="conv12")
        self.conv13 = tf.layers.Conv1D(filters=1024, kernel_size=3, padding="valid", name="conv13")
        self.conv14 = tf.layers.Conv1D(filters=1024, kernel_size=3, padding="valid", name="conv14")
        self.conv15 = tf.layers.Conv1D(filters=1024, kernel_size=3, padding="valid", name="conv15")
        self.conv16 = tf.layers.Conv1D(filters=1024, kernel_size=3, padding="valid", name="conv16")

        self.maxpool1 = tf.layers.MaxPooling1D(pool_size=3, strides=3, name="maxpool1")
        self.maxpool2 = tf.layers.MaxPooling1D(pool_size=3, strides=3, name="maxpool2")
        self.maxpool3 = tf.layers.MaxPooling1D(pool_size=3, strides=3, name="maxpool3")

        self.flatten = tf.layers.Flatten(name="flatten")
        self.dropout = tf.layers.Dropout(0.5, name="dropout")

        self.fc1 = tf.layers.Dense(256, activation=tf.nn.relu, name="fc1")
        self.fc2 = tf.layers.Dense(128, activation=tf.nn.relu, name="fc2")
        self.out = tf.layers.Dense(self.out_dim, name="out")

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # global step
        self.global_step = 0

    def predict(self, X, training):
        X = tf.one_hot(X, self.num_chars, axis=2)

        x = self.conv11(X)
        x = self.maxpool1(x)
        x = self.conv12(x)
        x = self.maxpool2(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        if training:
            x = self.dropout(x)

        x = self.fc2(x)
        if training:
            x = self.dropout(x)

        pred = self.out(x)
        return pred

    def call(self, X, training):
        return self.predict(X, training)

    def loss(self, X, y, training):
        """calculate loss of the batch
        Args:
            X : input tensor
            y : target label
            training : whether apply dropout or not
        """
        prediction = self.predict(X, training)
        loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction)
        loss_val = tf.reduce_sum(loss_val)
        loss_val += tf.nn.l2_loss(self.out.weights[0])

        return loss_val, prediction

    def grad(self, X, y, training):
        with tfe.GradientTape() as tape:
            loss_val, _ = self.loss(X, y, training)
        return tape.gradient(loss_val, self.variables), loss_val

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
                for X, y in tqdm_wrapper(dataset_train, total=batchlen_train, desc="TRAIN %3s" % self.global_step):
                    grads, batch_loss = self.grad(X, y, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                epoch_loss_val = 0.0
                val_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_val, total=batchlen_val, desc="VAL   %3s" % self.global_step):
                    batch_loss, pred = self.loss(X, y, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)
                    pred = tf.argmax(pred, axis=1)
                    pred = tf.cast(pred, y.dtype)
                    val_accuracy(pred, y)

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
        # init
        self.call(tf.convert_to_tensor(np.zeros([1, self.in_dim]), dtype="int64"), True)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
