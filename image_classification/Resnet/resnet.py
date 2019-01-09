import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from blocks_resnet import IdentitiyBlock_3, ConvolutionBlock_3
from tqdm import tqdm, tqdm_notebook
from colorama import Fore, Style


class Resnet(tf.keras.Model):
    """ Resnet model for CIFAR-10 dataset.
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
        super(Resnet, self).__init__()
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

        self.iden3b = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])
        self.iden3c = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])
        self.iden3d = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])

        self.conv4a = ConvolutionBlock_3(filters=[32, 32, 64], kernel_sizes=[(1, 1), (3, 3), (1, 1), (1, 1)])
        self.iden4b = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4c = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4d = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4e = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4f = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])

        self.conv5a = ConvolutionBlock_3(filters=[64, 64, 128], kernel_sizes=[(1, 1), (3, 3), (1, 1), (1, 1)])
        self.iden5b = IdentitiyBlock_3([64, 64, 128], [(1, 1), (3, 3), (1, 1)])
        self.iden5c = IdentitiyBlock_3([64, 64, 128], [(1, 1), (3, 3), (1, 1)])

        self.avgpool = tf.layers.AveragePooling2D((7, 7), (1, 1))

        self.flatten = tf.layers.Flatten()

        self.out_layer = tf.layers.Dense(out_dim)

        # optimizer
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

        x = self.iden3b(x, training=training)
        x = self.iden3c(x, training=training)
        x = self.iden3d(x, training=training)

        x = self.conv4a(x, training=training)
        x = self.iden4b(x, training=training)
        x = self.iden4c(x, training=training)
        x = self.iden4d(x, training=training)
        x = self.iden4e(x, training=training)
        x = self.iden4f(x, training=training)

        x = self.conv5a(x, training=training)
        x = self.iden5b(x, training=training)
        x = self.iden5c(x, training=training)

        x = self.avgpool(x)
        x = self.out_layer(self.flatten(x))

        return x

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

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        dummy_input = tf.constant(tf.zeros((1,) + self.input_dim))
        dummy_pred = self.predict(dummy_input, False)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
