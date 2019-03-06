import os
import re
import time
import tensorflow as tf


class Logger(object):
    def __init__(self, sess, path, saver):
        self.sess = sess
        self.path = path
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.path, "train"), self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.path, "eval"))
        self.last_saver = saver

    def summarize(self, epoch, tag_value, tag_name):
        loss_sum = tf.Summary()
        loss_sum.value.add(tag=tag_name, simple_value=tag_value)
        self.test_summary_writer.add_summary(loss_sum, epoch)
        self.test_summary_writer.flush()

    # load latest checkpoint from the experiment path defined in the config file
    def load_last_ckpt(self, sess):
        print("Checking path {} ...".format(os.path.join(self.path, "train")))
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.path, "train"))
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.last_saver.restore(sess, latest_checkpoint)
            print("-----------------Model loaded")
            return re.search(r'model.ckpt-(\d{1,})', latest_checkpoint).group(1)
        else:
            print("No model found")
            return 'null'

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, step):
        t0 = time.time()
        print("Saving model to {} ...".format(os.path.join(self.path, "train")))
        save_path = os.path.join(self.path, "train", 'model')
        self.last_saver.save(sess, save_path, step)
        print("Model saved")
        print("Using time: ", time.time() - t0)
