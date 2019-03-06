import os
import random
import importlib
import time
import numpy as np
import tensorflow as tf
import texar as tx
from model_attdec import AttentionDecoder
from model_attcovdec import CovAttentionDecoder
from model_encattdec import EncAttentionDecoder
from model_retrieval import RetrievalAttDecoder
from model_sen import EncAttentionSenDecoder
from model_encretrieval import EncRetrievalAttDecoder
from model_encretrieval_test import EncRetrievalAttDecoderTest
from model_encretrieval_cov import EncRetrievalAttDecoderCov
from model_encretrieval_word import EncRetrievalAttDecoderWord
from model_encretrieval_word_enc import EncRetrievalAttDecoderWordEnc
from model_pnn import PNN
from model_encretrieval_word_enc_copy import EncRetrievalAttDecoderWordEncCopy
from model_encretrieval_word_enc_add import EncRetrievalAttDecoderWordEncAdd
from train import Trainer
from generation import Generator
from utils.logger import Logger
from utils import util

flags = tf.flags
flags.DEFINE_string("mode", "train", "The running mode (train, decode)")
flags.DEFINE_string("config", "config", "The model config.")
flags.DEFINE_string("exp_name", "path", "The exp path")
flags.DEFINE_string("method_name", "name", "The model class name")
# flags.DEFINE_string("config", "config_retrieval", "The model config.")
# flags.DEFINE_string("config", "config_hier", "The model config.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)
config.mode = FLAGS.mode
if FLAGS.exp_name != 'path':
    config.exp_name = FLAGS.exp_name
if FLAGS.method_name != 'name':
    config.method_name = FLAGS.method_name


def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.DEBUG)  # choose what level of logging you want
    tf.logging.info('Load model from %s and %s ...', FLAGS.config, config.method_name)
    tf.logging.info('Starting model in %s mode in %s ...', config.mode, config.exp_name)

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    config.log_root = os.path.join(config.log_root, config.exp_name)
    if not os.path.exists(config.log_root):
        if config.mode == "train":
            os.makedirs(config.log_root)

            train_dir = os.path.join(config.log_root, "train")
            if not os.path.exists(train_dir): os.makedirs(train_dir)
            eval_dir = os.path.join(config.log_root, "eval")
            if not os.path.exists(eval_dir): os.makedirs(eval_dir)
        # else:
        #     raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % FLAGS.log_root)

    tf.set_random_seed(111)  # a seed value for randomness
    random.seed(111)
    initializer = tf.random_uniform_initializer(-config.rand_unif_init, config.rand_unif_init)

    if config.mode == 'train':
        # create your data generator
        train_data = tx.data.MultiAlignedData(config.train_data_hparams)
        val_data = tx.data.MultiAlignedData(config.val_data_hparams)
        data_loader = tx.data.TrainTestDataIterator(train=train_data, val=val_data)
        data_batch = data_loader.get_next()

        vocab = train_data.vocab('x')
        pre_word2vec = None
        # pre_word2vec = train_data.embedding_init_value(0).word_vecs

        # create instance of the model you want
        tf.logging.info("Start building graph")
        t0 = time.time()
        with tf.variable_scope("model", initializer=initializer):
            # model = CovAttentionDecoder(config, data_batch, vocab, pre_word2vec)
            # model = AttentionDecoder(config, data_batch, vocab, pre_word2vec)
            # model = EncAttentionDecoder(config, data_batch, vocab, pre_word2vec)
            # model = RetrievalAttDecoder(config, data_batch, vocab, pre_word2vec)
            # model = EncAttentionSenDecoder(config, data_batch, vocab, pre_word2vec)
            model = globals()[config.method_name](config, data_batch, vocab, pre_word2vec)

        tf.logging.info("Build graph in %2f second", (time.time() - t0))
        util.count_parameter()

        # create tensorflow session
        saver = tf.train.Saver(max_to_keep=1)
        sv = tf.train.Supervisor(logdir=os.path.join(config.log_root, "train"),
                                 is_chief=True,
                                 saver=saver,
                                 save_model_secs=60 * 5,  # checkpoint every 60 secs
                                 summary_op=None,
                                 summary_writer=None,
                                 save_summaries_secs=0,  # save summaries for tensorboard every 60 secs
                                 global_step=model.global_step)
        with sv.prepare_or_wait_for_session(config=util.get_config()) as sess:
        # with tf.Session(config=util.get_config()) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(tf.local_variables_initializer())
        #     sess.run(tf.tables_initializer())

            # epoch = model.load(sess)
            # create tensorboard logger
            logger = Logger(sess, config.log_root, saver)

            # create trainer and path all previous components to it
            trainer = Trainer(sess, model, data_loader, config, logger, vocab)
            trainer.train(0)
            # trainer.train(int(epoch) + 1)
    elif config.mode == 'decode':
        # create your data generator
        test_data = tx.data.MultiAlignedData(config.test_data_hparams)
        data_loader = tx.data.TrainTestDataIterator(test=test_data)
        data_batch = data_loader.get_next()

        vocab = test_data.vocab('x')

        # create instance of the model you want
        with tf.variable_scope("model", initializer=initializer):
            # model = CovAttentionDecoder(config, data_batch, vocab)
            # model = AttentionDecoder(config, data_batch, vocab)
            # model = EncAttentionDecoder(config, data_batch, vocab)
            # model = RetrievalAttDecoder(config, data_batch, vocab)
            # model = EncAttentionSenDecoder(config, data_batch, vocab)
            model = globals()[config.method_name](config, data_batch, vocab)

        util.count_parameter()
        print(config.log_root)

        # create tensorflow session
        with tf.Session(config=util.get_config()) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            # data_loader.switch_to_test_data(sess)
            # results = model.run_test_shuffle(sess)
            # for ii in results:
            #
            #     print(ii, results[ii])
            # output_ids = results['para']
            # output_texts = tx.utils.map_ids_to_strs(
            #     ids=output_ids, vocab=vocab)
            # output_texts = tx.utils.str_join(output_texts)
            # print(output_texts[0])
            # output_ids = results['para_shuf']
            # output_texts = tx.utils.map_ids_to_strs(
            #     ids=output_ids, vocab=vocab)
            # output_texts = tx.utils.str_join(output_texts)
            # print(output_texts[0])
            # data_batch_ = sess.run(data_batch)
            # print(data_batch_)
            # target_text = data_batch_['x_text']
            # target_texts = tx.utils.str_join(target_text)

            # output_text = data_batch_['z_text']
            # output_texts = tx.utils.str_join(output_text)
            #
            # re_text = data_batch_['yy_text']
            # re_texts = tx.utils.str_join(re_text)
            # re_texts = np.reshape(re_texts, [config.batch_size, -1])
            # re_texts = tx.utils.str_join(re_texts)
            #
            # # Writes samples
            # tx.utils.write_paired_text(target_texts, output_texts, os.path.join(config.log_root, 'datacheck'), append=True, mode='v')
            # tx.utils.write_paired_text(target_texts, re_texts, os.path.join(config.log_root, 'datacheck2'), append=True, mode='v')

            epoch = model.load(sess)

            # create trainer and path all previous components to it
            generator = Generator(sess, model, data_loader, config, vocab)
            generator.generate(epoch)
    else:
        raise ValueError("The 'mode' flag must be one of train/decode")


if __name__ == '__main__':
    tf.app.run()
