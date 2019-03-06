import time
import os
import re
import tensorflow as tf
import texar as tx
from utils.decoders import get_max_loop_function
from utils.decoders import apart_attention_decoder_word_one
from utils.decoders import _mask_memory


class EncRetrievalAttDecoderWordEnc(object):
    def __init__(self, config, data_batch, vocab, pre_word2vec=None):
        self.config = config
        self.data_batch = data_batch
        self.vocab = vocab
        self.pre_word2vec = pre_word2vec

        # save attribute .. NOTE DON'T FORGET TO CONSTRUCT THE SAVER ON YOUR MODEL
        self.saver = None
        self.build_model()
        self.init_saver()

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, epoch):
        print("Saving model to {} ...".format(self.config.log_root))
        save_path = os.path.join(self.config.log_root, 'model')
        self.saver.save(sess, save_path, epoch)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        print("Checking path {} ...".format(self.config.log_root))
        latest_checkpoint = tf.train.latest_checkpoint(self.config.log_root)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("-----------------Model loaded")
            return re.search(r'model-(\d{1,})', latest_checkpoint).group(1)
        else:
            print("No model found")
            exit(0)

    def build_model(self):
        # input data
        src_text_ids = self.data_batch['z_text_ids']  # batch_size * (max_enc + 2)(include BOS and EOS)
        src_text_length = self.data_batch['z_length']  # batch_size

        tgt_text_ids = self.data_batch['x_text_ids']  # batch_size * max_words(dynamic shape)
        tgt_text_length = self.data_batch['x_length']  # batch_size

        sentence_text_ids = self.data_batch['yy_text_ids']
        sentence_text_length = self.data_batch['yy_length']
        sentence_utterance_cnt = self.data_batch['yy_utterance_cnt']

        self._src_text = self.data_batch['z_text']
        self._tgt_text = self.data_batch['x_text']

        def get_a_cell():
            LSTM_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim, forget_bias=0.0)
            if self.config.mode == 'train' and self.config.keep_prob < 1:
                LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell, output_keep_prob=self.config.keep_prob)
            return LSTM_cell

        if self.pre_word2vec is None:
            source_embedder = tx.modules.WordEmbedder(
                vocab_size=self.vocab.size, hparams=self.config.embedder)
        else:
            source_embedder = tx.modules.WordEmbedder(init_value=self.pre_word2vec)

        encoder = tx.modules.BidirectionalRNNEncoder(
            hparams=self.config.encoder)

        enc_outputs, _ = encoder(inputs=source_embedder(src_text_ids), sequence_length=src_text_length)
        enc_outputs = tf.concat(enc_outputs, axis=2)

        # convert_layer = tf.layers.Dense(800, activation=tf.nn.tanh, use_bias=True)
        re_sen_num = tf.shape(sentence_text_ids)[1]
        re_word_num = tf.shape(sentence_text_ids)[2]
        sentence_emb = source_embedder(sentence_text_ids)
        sentence_emb = tf.reshape(sentence_emb, [self.config.batch_size * re_sen_num, re_word_num, self.config.emb_dim])

        re_encoder = tx.modules.BidirectionalRNNEncoder(
            hparams=self.config.re_encoder)

        re_enc_outputs, _ = re_encoder(inputs=sentence_emb, sequence_length=tf.reshape(sentence_text_length, [-1]))
        re_enc_outputs = tf.concat(re_enc_outputs, axis=2)
        re_enc_outputs = tf.reshape(re_enc_outputs, [self.config.batch_size, re_sen_num, re_word_num, self.config.hidden_dim])

        # decode layer
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell() for _ in range(self.config.num_layers)]
        )
        initial_state = cell.zero_state(self.config.batch_size, tf.float32)

        # out projection
        output_layer = tf.layers.Dense(units=self.vocab.size, activation=None, use_bias=True)

        loop_function = None
        if self.config.mode == 'decode':
            startids = tf.ones_like(src_text_length) * self.vocab.bos_token_id
            loop_function = get_max_loop_function(source_embedder.embedding, startids, output_layer)

        dec_inputs = source_embedder(tgt_text_ids[:, :-1])
        if self.config.mode == 'train' and self.config.keep_prob < 1:
            dec_inputs = tf.nn.dropout(dec_inputs, self.config.keep_prob)

        outputs, logits = apart_attention_decoder_word_one(
            cell, initial_state, enc_outputs, src_text_length, re_enc_outputs, sentence_text_length, sentence_utterance_cnt, output_layer,
            dec_inputs, tgt_text_length - 1, self.config, loop_function
        )

        self._predict_ids = tf.argmax(logits, axis=-1)

        # loss
        self.mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_text_ids[:, 1:],
            logits=logits,
            sequence_length=tgt_text_length - 1,
        )

        # self.mle_loss = tf.check_numerics(self.mle_loss, 'loss is nan')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # with tf.control_dependencies([self.mle_loss]):
        self._train_op = tx.core.get_train_op(self.mle_loss, global_step=self.global_step, hparams=self.config.opt)

        loss_summary = tf.summary.scalar('loss', self.mle_loss)
        self._summaries = tf.summary.merge([loss_summary])

        t_vars = tf.trainable_variables()
        var_summary = [tf.summary.histogram("{}".format(v.name), v) for v in t_vars]
        self._var_summaries = tf.summary.merge(var_summary)

    def run_train_step(self, sess):
        feed_dict = {}
        feed_dict[tx.global_mode()] = tf.estimator.ModeKeys.TRAIN
        to_return = {
            'train_op': self._train_op,
            'loss': self.mle_loss,
            'global_step': self.global_step,
            'summaries': self._summaries,
        }
        return sess.run(to_return, feed_dict)

    def run_var_check(self, sess):
        feed_dict = {}
        to_return = {
            'summaries': self._var_summaries,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess):
        feed_dict = {}
        feed_dict[tx.global_mode()] = tf.estimator.ModeKeys.EVAL
        to_return = {
            'loss': self.mle_loss,
        }
        return sess.run(to_return, feed_dict)

    def run_generate_step(self, sess):
        feed_dict = {}
        feed_dict[tx.global_mode()] = tf.estimator.ModeKeys.EVAL
        to_return = {
            'output_ids': self._predict_ids,
            'target_text': self._tgt_text,
            'input_text': self._src_text,
        }
        return sess.run(to_return, feed_dict)

    def run_data_step(self, sess):
        feed_dict = {}
        to_return = {
            'target_text': self._tgt_text,
            'input_text': self._src_text,
        }
        return sess.run(to_return, feed_dict)


