import numpy as np
import time
import os
import tensorflow as tf
import texar as tx


class Generator(object):
    def __init__(self, sess, model, data_loader, config, vocab):
        self.sess = sess
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.vocab = vocab

    def generate(self, epoch):
        self.data_loader.switch_to_test_data(self.sess)
        step = 0
        refs, hypos = [], []
        while True:
            try:
                # for step in range(len(self.data_loader)):
                t0 = time.time()
                results = self.generate_step()
                # loss = results['loss']

                input_text = results['input_text']
                input_texts = tx.utils.strip_special_tokens(
                    input_text, is_token_list=True)

                target_text = results['target_text']
                target_texts = tx.utils.strip_special_tokens(
                    target_text, is_token_list=True)
                target_texts = tx.utils.str_join(target_texts)

                output_ids = results['output_ids']
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids, vocab=self.vocab)

                if self.config.exp_name.__contains__('hier'):
                    output_texts = np.reshape(output_texts, [self.config.batch_size, -1])
                    output_texts = tx.utils.str_join(output_texts)
                    output_texts = [text.split(' <PAD>')[0] if text.__contains__(' <PAD>') else text for text in output_texts]

                    target_texts = np.reshape(target_texts, [self.config.batch_size, -1])
                    target_texts = tx.utils.str_join(target_texts)

                for hypo, ref in zip(output_texts, target_texts):
                    hypos.append(hypo)
                    refs.append([ref])

                # Writes samples
                tx.utils.write_paired_text(tx.utils.str_join(input_texts), output_texts, os.path.join(self.config.log_root, epoch + self.config.sample_path), append=True, mode='v')
                tx.utils.write_paired_text(target_texts, output_texts, os.path.join(self.config.log_root, epoch + self.config.sample_path + '2'), append=True, mode='v')

                if step % 100 == 0:
                    # tf.logging.info("%s Testing loss : %2f, sec/batch : %2f" % (step, loss, (time.time() - t0)))
                    print(' '.join(input_texts[0]))
                    # print(' '.join(topic_texts[0]))
                    print(output_texts[0])

                step += 1
            except tf.errors.OutOfRangeError:
                break

        test_bleu = tx.evals.corpus_bleu(list_of_references=refs, hypotheses=hypos, max_order=2, return_all=True)
        print('=' * 50)
        print_str = 'BLEU={d[0]:.4f}, b1={d[1]:.4f}, b2={d[2]:.4f}'.format(d=test_bleu)
        print(print_str)
        print('=' * 50)
        with open(os.path.join(self.config.log_root, epoch + '.bleu'), 'w') as f:
            f.write(print_str)

    def generate_step(self):
        return self.model.run_generate_step(self.sess)
