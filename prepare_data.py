from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import open  # pylint: disable=redefined-builtin
import tensorflow as tf
import texar as tx

# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("data_path", "/home/wangw/data_research/essay", "Directory containing SST data. ")

FLAGS = flags.FLAGS


def transform_raw_sst(data_path, raw_fn, new_fn):
    """Transforms the raw data format to a new format.
    """
    fout_x = open(new_fn, 'w', encoding='utf-8')
    fout_y = open(new_fn + '.label', 'w', encoding='utf-8')
    fout_z = open(new_fn + '.unique', 'w', encoding='utf-8')
    # fout_w = open(new_fn + '.index', 'w', encoding='utf-8')
    fout_1 = open(new_fn + '.1', 'w', encoding='utf-8')
    fout_2 = open(new_fn + '.2', 'w', encoding='utf-8')
    fout_3 = open(new_fn + '.3', 'w', encoding='utf-8')
    fout_4 = open(new_fn + '.4', 'w', encoding='utf-8')
    fout_5 = open(new_fn + '.5', 'w', encoding='utf-8')

    fin_name = os.path.join(data_path, raw_fn)
    with open(fin_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.split('</d>')
            sent = line[0].strip()
            keywords = line[1].strip()
            fout_x.write(sent + '\n')
            fout_y.write(keywords + '\n')

            unique_words = list(set(sent.split()))
            fout_z.write(' '.join(unique_words) + '\n')

            word_dict = enumerate(unique_words)
            word_dict = dict((y, x) for x, y in word_dict)
            keywords = keywords.split()
            assert len(keywords) == 5
            indexs = [word_dict[w] for w in keywords]
            fout_1.write(unicode(indexs[0]) + '\n')
            fout_2.write(unicode(indexs[1]) + '\n')
            fout_3.write(unicode(indexs[2]) + '\n')
            fout_4.write(unicode(indexs[3]) + '\n')
            fout_5.write(unicode(indexs[4]) + '\n')

    return new_fn


def prepare_data(data_path):
    fn_train = transform_raw_sst(data_path, 'train.txt', './data/essay.train')
    transform_raw_sst(data_path, 'dev.txt', './data/essay.dev')
    transform_raw_sst(data_path, 'test.txt', './data/essay.test')

    vocab = tx.data.make_vocab(fn_train, max_vocab_size=50000, newline_token='')
    fn_vocab = './data/essay.vocab'
    with open(fn_vocab, 'w', encoding='utf-8') as f_vocab:
        for v in vocab:
            f_vocab.write(v + '\n')

    tf.logging.info('Preprocessing done: {}'.format(data_path))


def _main(_):
    prepare_data(FLAGS.data_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=_main)