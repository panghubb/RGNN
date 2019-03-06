import numpy as np
import time
import tensorflow as tf
import texar as tx
import os


class Trainer(object):
    def __init__(self, sess, model, data_loader, config, logger, vocab):
        self.sess = sess
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        self.vocab = vocab

    def train(self, start=0):
        tf.logging.info("Start training")
        for cur_epoch in range(start, self.config.max_epochs):

            t0 = time.time()
            epoch_loss = self.train_epoch(cur_epoch)
            tf.logging.info("Epoch : %s Training loss : %2f, sec/epoch : %2f" % (cur_epoch, epoch_loss, (time.time() - t0)))

            # We save the model at the end of each epoch
            if cur_epoch % self.config.model_save_freq == 0:
                tf.logging.info("Save model...")
                self.model.save(self.sess, cur_epoch)

            t0 = time.time()
            dev_loss = self.test_epoch(cur_epoch)
            tf.logging.info("Epoch : %s Testing loss : %2f, sec/epoch : %2f" % (cur_epoch, dev_loss, (time.time() - t0)))

            self.logger.summarize(cur_epoch, epoch_loss, "epoch/train_avg_loss")
            self.logger.summarize(cur_epoch, dev_loss, "epoch/eval_avg_loss")

    def train_epoch(self, cur_epoch):
        self.data_loader.switch_to_train_data(self.sess)
        losses = []
        step = 0
        while True:
            try:
                t0 = time.time()
                # if cur_epoch == 19 and step < 16493:
                #     self.model.run_data_step(self.sess)
                #     step += 1
                #     continue
                # if cur_epoch == 19:
                #     if step == 16493:
                #         # tf.logging.info("Save model...")
                #         # self.model.save(self.sess, step - 1)
                #
                #         results = self.model.run_generate_step(self.sess)
                #         # results = self.model.run_generate_step(self.sess)
                #         input_text = results['input_text']
                #
                #         target_text = results['target_text']
                #         target_texts = tx.utils.str_join(target_text)
                #
                #         output_ids = results['output_ids']
                #         output_texts = tx.utils.map_ids_to_strs(
                #             ids=output_ids, vocab=self.vocab)
                #
                #         # Writes samples
                #         tx.utils.write_paired_text(tx.utils.str_join(input_text), output_texts, os.path.join(self.config.log_root, 'train', str(cur_epoch) + str(step) + self.config.sample_path), append=True, mode='v')
                #         tx.utils.write_paired_text(target_texts, output_texts, os.path.join(self.config.log_root, 'train', str(cur_epoch) + str(step) + self.config.sample_path + '2'), append=True, mode='v')

                # for known error, catch and continue to train
                # try:
                results = self.train_step()
                loss = results['loss']
                # except tf.errors.InvalidArgumentError:
                #     print(step)
                #     self.logger.train_summary_writer.flush()
                #     results = self.model.run_var_check(self.sess)
                #     self.logger.train_summary_writer.add_summary(results['summaries'], results['global_step'])
                #     self.logger.train_summary_writer.flush()
                #     print("Loss is not finite. Recording epoch and step.")
                #     with open(os.path.join(self.config.log_root, 'errorlog'), 'a') as f:
                #         f.write(str(cur_epoch) + str(step) + '\n')
                #     continue

                # for unknown error, check variable and stop
                if not np.isfinite(loss):
                    print(step, loss)
                    # self.logger.train_summary_writer.flush()
                    # results = self.model.run_var_check(self.sess)
                    # self.logger.train_summary_writer.add_summary(results['summaries'], results['global_step'])
                    # self.logger.train_summary_writer.flush()
                    # raise Exception("Loss is not finite. Stopping.")

                    # restore last ckpt
                    with open(os.path.join(self.config.log_root, 'errorlog'), 'a') as f:
                        f.write(str(cur_epoch) + str(step) + '\n')
                    self.logger.load_last_ckpt(self.sess)
                    continue
                losses.append(loss)

                # get the summaries and iteration number so we can write summaries to tensorboard
                summaries = results['summaries']  # we will write these summaries to tensorboard using summary_writer
                train_step = results['global_step']  # we need this to update our running average loss

                self.logger.train_summary_writer.add_summary(summaries, train_step)
                if step % 1000 == 0:
                    tf.logging.info("Epoch : %s/%s Training loss : %2f, sec/batch : %2f" % (cur_epoch, step, loss, (time.time() - t0)))

                if train_step % 100 == 0:  # flush the summary writer every so often
                    self.logger.train_summary_writer.flush()

                step += 1
            except tf.errors.OutOfRangeError:
                break

        loss = np.mean(losses)
        return loss

    def train_step(self):
        return self.model.run_train_step(self.sess)

    def test_epoch(self, cur_epoch):
        self.data_loader.switch_to_val_data(self.sess)
        losses = []
        step = 0
        while True:
            try:
                t0 = time.time()
                results = self.test_step()
                loss = results['loss']
                losses.append(loss)

                if step % 1000 == 0:
                    tf.logging.info("Epoch : %s/%s Testing loss : %2f, sec/batch : %2f" % (cur_epoch, step, loss, (time.time() - t0)))

                step += 1
            except tf.errors.OutOfRangeError:
                break

        final_loss = np.mean(losses)
        return final_loss

    def test_step(self):
        return self.model.run_eval_step(self.sess)
