from __future__ import absolute_import, division, print_function
import os
from collections import deque
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from modules import get_position_encoding, MultiHeadAttention, FeedForward, PrePostProcessingWrapper
from colorama import Fore, Style
import time


class Transformer(tf.keras.Model):
    """ Transformer Main Model for Translation Task
    Args:
        hidden_size: base size of Tensors. maintained for all sub modules
        num_heads: number of heads(for multi-head attention)
        dropout_rate: dropout ratio
        input_length: length of input sequence include paddings
        target_length: length of target sequence include paddings
        input_wordcnt: number of vocabulary in input language
        target_wordcnt: number of vocabulary in target language
        layer_depth: number of loop for Transformer logic
        learning_rate: for optimizer, default is 1e-3
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning, default is cpu:0
    """

    def __init__(self, hidden_size, num_heads, dropout_rate, input_length, target_length, input_wordcnt, target_wordcnt,
                 layer_depth, learning_rate=1e-3, device_name="cpu:0", checkpoint_directory="ckpt/"):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.input_length = input_length
        self.target_length = target_length
        self.input_wordcnt = input_wordcnt
        self.target_wordcnt = target_wordcnt
        self.layer_depth = layer_depth
        self.device_name = device_name

        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)

        self.embedding_input = tf.keras.layers.Embedding(self.input_wordcnt, hidden_size, name="embedding_input")
        self.embedding_target = tf.keras.layers.Embedding(self.target_wordcnt, hidden_size, name="embedding_target")

        self.enc_self_attention_layers = list()
        self.enc_ffn_layers = list()
        for i in range(layer_depth):
            attention_self_enc = MultiHeadAttention(self.hidden_size, self.num_heads, name="attention_self_enc_%s" % i)
            attention_self_enc = PrePostProcessingWrapper(attention_self_enc, self.hidden_size, self.dropout_rate,
                                                          name="wrapped_attention_self_enc_%s" % i)
            ffn = FeedForward(self.hidden_size, self.hidden_size * 4, name="ffn_enc_%s" % i)
            ffn = PrePostProcessingWrapper(ffn, self.hidden_size, self.dropout_rate, name="wrapped_ffn_enc_%s" % i)
            self.enc_self_attention_layers.append(attention_self_enc)
            self.enc_ffn_layers.append(ffn)

        self.dec_self_attention_layers = list()
        self.dec_attention_layers = list()
        self.dec_ffn_layers = list()
        for i in range(layer_depth):
            attention_self_dec = MultiHeadAttention(self.hidden_size, self.num_heads, name="attention_self_dec_%s" % i)
            attention_self_dec = PrePostProcessingWrapper(attention_self_dec, self.hidden_size, self.dropout_rate,
                                                          name="wrapped_attention_self_dec_%s" % i)
            attention_vanilla = MultiHeadAttention(self.hidden_size, self.num_heads,
                                                   name="attention_vanilla_dec_%s" % i)
            attention_vanilla = PrePostProcessingWrapper(attention_vanilla, self.hidden_size, self.dropout_rate,
                                                         name="wrapped_attention_vanilla_dec_%s" % i)
            ffn = FeedForward(self.hidden_size, self.hidden_size * 4, name="ffn_dec_%s" % i)
            ffn = PrePostProcessingWrapper(ffn, self.hidden_size, self.dropout_rate, name="wrapped_ffn_dec_%s" % i)
            self.dec_self_attention_layers.append(attention_self_dec)
            self.dec_attention_layers.append(attention_vanilla)
            self.dec_ffn_layers.append(ffn)

        self.out = tf.layers.Dense(target_wordcnt, name="out")

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.997, epsilon=1e-09)

        # logging
        self.global_step = 0

        # early_stopping
        self.early_stopping_lookup = deque()

    def encoding_stack(self, x_enc, train):
        x = self.embedding_input(x_enc)
        x += get_position_encoding(tf.shape(x)[1], self.hidden_size)
        for i in range(self.layer_depth):
            x = self.enc_self_attention_layers[i](x, x, train=train)
            x = self.enc_ffn_layers[i](x, train=train)
        return x

    def decoding_stack(self, x_dec, out_enc, train):
        x = self.embedding_target(x_dec)
        x += get_position_encoding(tf.shape(x)[1], self.hidden_size)
        for i in range(self.layer_depth):
            x = self.dec_self_attention_layers[i](x, x, train=train, masking=True)
            x = self.dec_attention_layers[i](x, out_enc, train=train)
            x = self.dec_ffn_layers[i](x, train=train)
        return x

    def call(self, x_enc, x_dec, train):
        out_enc = self.encoding_stack(x_enc, train=train)
        out_dec = self.decoding_stack(x_dec, out_enc, train=train)
        out = self.out(out_dec)
        return out

    # https://github.com/tensorflow/models/blob/master/official/transformer/utils/metrics.py
    def label_smoothing(self, labels, smoothing):
        confidence = 1.0 - smoothing
        low_confidence = (1.0 - confidence) / tf.to_float(self.target_wordcnt - 1)
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=self.target_wordcnt,
            on_value=confidence,
            off_value=low_confidence)
        return soft_targets

    def loss(self, x_enc, x_dec, x_dec_shifted, train):
        logits = self.call(x_enc, x_dec_shifted, train)
        # label smoothing(0.1)
        soft_targets = self.label_smoothing(x_dec, 0.1)
        loss_val = tf.nn.softmax_cross_entropy_with_logits_v2(labels=soft_targets, logits=logits)
        # masking for padding
        loss_val *= tf.cast(tf.sign(tf.abs(x_dec)), tf.float32)
        return loss_val

    def grad(self, x_enc, x_dec, x_dec_shifted, train):
        with tfe.GradientTape() as tape:
            loss_val = self.loss(x_enc, x_dec, x_dec_shifted, train)
        return tape.gradient(loss_val, self.variables), loss_val

    def fit(self, X_train, y_train, X_val, y_val, bos_index=2, epochs=1, verbose=1, batch_size=32, saving=False,
            tqdm_option=None):
        """train the network
        Args:
            X_train : train dataset input
            y_train : train dataset label
            X_val : validation dataset input
            y_val : validation dataset input
            bos_index : "<BOS>" token's index in target vocabulary
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
                for X, y in tqdm_wrapper(dataset_train, total=batchlen_train, desc="TRAIN%3s" % self.global_step):
                    y_shifted = tf.pad(y, [[0, 0], [1, 0]], constant_values=bos_index)[:, :-1]
                    grads, batch_loss = self.grad(X, y, y_shifted, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                epoch_loss_val = 0.0
                for X, y in tqdm_wrapper(dataset_val, total=batchlen_val, desc="VAL  %3s" % self.global_step):
                    y_shifted = tf.pad(y, [[0, 0], [1, 0]], constant_values=bos_index)[:, :-1]
                    batch_loss = self.loss(X, y, y_shifted, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss / batchlen_train))
                    print("VAL   loss   : %.4f" % (epoch_loss_val / batchlen_val))

                    if saving:
                        self.save()
                    print("=" * 25 + Style.RESET_ALL)
                time.sleep(1)

    def predict(self, x_enc, start_token, eos_index=3, beam_cnt=1):
        """apply n-beam search for inference
        Args:
            x_enc : input sentence
            start_token : initial target sentence with only <BOS> token
            eos_index : index of <EOS> token in target vocabulary
            beam_cnt : beam-search option , if 1 it is same as greedy search
        Returns:
            beams with highest probability.
        """
        def end_condition(beamlist, last):
            for beam_ in beamlist:
                if beam_[0][last] != eos_index:
                    return False
            return True

        enc_out = self.encoding_stack(x_enc, False)

        beams = [start_token]
        now_len = 0  # length except <BOS> token
        finished = []
        logprobs = [0.0]

        while (now_len+1 < self.target_length) and (not end_condition(beams, now_len)):
            candidate_beam = []
            for beam_idx, beam in enumerate(beams):
                # if beam got <EOS>, move to finished and continue
                if tf.equal(beam[0][now_len], 3):
                    finished.append((logprobs[beam_idx], beam))
                    continue
                dec_out = self.decoding_stack(beam, enc_out, False)
                logit = tf.nn.softmax(self.out(dec_out))
                probs_candidate, next_indexes_candidate = tf.nn.top_k(logit[:, now_len], k=beam_cnt, sorted=True)
                logprobs_candidate = tf.log(probs_candidate)[0]
                next_indexes_candidate = tf.reshape(next_indexes_candidate, [-1, 1])
                # make adding arrays, [0, 0, ....{now_len: new index}, 0, 0, ...]
                adding_arrays = tf.pad(next_indexes_candidate, [[0, 0], [now_len + 1, self.target_length - now_len-2]],
                                       constant_values=0)
                candidate_beams = [beam + adding_array for adding_array in adding_arrays]

                for candidate_index, logprob_candidate in enumerate(logprobs_candidate):
                    candidate_beam.append((logprob_candidate + logprobs[beam_idx], candidate_beams[candidate_index]))

            picked_candidate = list(reversed(sorted(candidate_beam)))[:beam_cnt]
            logprobs = [logprob for logprob, _ in picked_candidate]
            beams = [beam for _, beam in picked_candidate]
            now_len += 1

        return finished + [(logprobs[i], beams[i]) for i in range(len(beams))]

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        dummy_input_enc = tf.zeros([1, self.input_length])
        dummy_input_dec = tf.zeros([1, self.target_length])
        dummy_pred = self.call(dummy_input_enc, dummy_input_dec, True)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
