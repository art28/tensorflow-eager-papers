import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from blocks_googlenet import InceptionBlock
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style


class GoogLEnet(tf.keras.Model):
    """ GoogLEnet model for CIFAR-10 dataset.
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
        super(GoogLEnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # layer declaration

        # first convolution is skipped because image size is already small
        # self.conv1 = tf.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2, 2), padding="same",
        #                               activation=tf.nn.relu)
        # self.maxpool1 =tf.layers.MaxPooling2D((3,3),(2,2), padding="same")

        self.conv2 = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool2 = tf.layers.MaxPooling2D((5, 5), (1, 1))  # this is custom pool to make size 28x28

        self.inception3a = InceptionBlock(conv11=8, reduce_conv33=12, conv33=16, reduce_conv55=2, conv55=4, convpool=2)
        self.inception3b = InceptionBlock(16, 16, 24, 4, 12, 8)
        self.maxpool3 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.inception4a = InceptionBlock(24, 12, 26, 2, 6, 8)
        self.inception4b = InceptionBlock(20, 14, 28, 3, 8, 8)
        self.inception4c = InceptionBlock(16, 16, 32, 3, 8, 8)
        self.inception4d = InceptionBlock(14, 18, 32, 4, 8, 8)
        self.inception4e = InceptionBlock(32, 20, 40, 4, 16, 16)
        self.maxpool4 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.inception5a = InceptionBlock(32, 20, 40, 4, 16, 16)
        self.inception5b = InceptionBlock(48, 24, 48, 6, 16, 16)
        self.avgpool = tf.layers.AveragePooling2D((7, 7), (1, 1))

        self.flatten = tf.layers.Flatten()
        self.dropout = tf.layers.Dropout(0.4)

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
        x = self.conv2(X)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = self.flatten(x)

        if training:
            x = self.dropout(x)

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
