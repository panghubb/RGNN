# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file defines the decoder"""
import numpy as np
import tensorflow as tf


def basic_decoder(cell, initial_state, memory, output_layer,
                      dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("basic_decoder", reuse=tf.AUTO_REUSE):

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, memory], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def attention_decoder(cell, initial_state, memory, memory_sequence_length, output_layer,
                      dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("attention_decoder", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("attention_v", [hidden_dim])
        memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
        query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

        memory = _mask_memory(memory, memory_sequence_length)
        meory_keys = memory_linear(memory)
        mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

        def attention(query, keys, values, mask):
            """Put attention masks on hidden using hidden_features and query."""
            with tf.variable_scope("Bahdanau_Attention"):
                y = query_linear(query)
                y = tf.reshape(y, [-1, 1, hidden_dim])
                # Attention mask is a softmax of v^T * tanh(...).
                s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                s = _maybe_mask_score(s, mask)
                aa = tf.nn.softmax(s)

                # Now calculate the attention-weighted vector d.
                d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                d = tf.reshape(d, [batch_size, -1])
            return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, mt], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, output_layer,
                      dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            re_memory_val_linear = tf.layers.Dense(units=hidden_dim * 2, activation=None, use_bias=True, name="value_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")
            re_gate = tf.layers.Dense(units=hidden_dim * 2, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            # retrieval attention
            retrieval_memory = _mask_memory(retrieval_memory, re_seq_length)
            re_memory_keys = re_memory_linear(retrieval_memory)
            # re_memory_values = re_memory_val_linear(retrieval_memory)
            re_score_mask = tf.sequence_mask(re_seq_length, maxlen=tf.shape(retrieval_memory)[1])

            def retrieval_attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(re_v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)
            retrieval_context, _ = retrieval_attention(word_output_state_, re_memory_keys, retrieval_memory, re_score_mask)
            retrieval_context = retrieval_context * re_gate(tf.concat([word_output_state_, retrieval_context], axis=1))

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, mt, retrieval_context], axis=1)
            # inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder_add(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, output_layer,
                      dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            re_memory_val_linear = tf.layers.Dense(units=hidden_dim * 2, activation=None, use_bias=True, name="value_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")
            re_gate = tf.layers.Dense(units=hidden_dim * 2, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            # retrieval attention
            retrieval_memory = _mask_memory(retrieval_memory, re_seq_length)
            re_memory_keys = re_memory_linear(retrieval_memory)
            # re_memory_values = re_memory_val_linear(retrieval_memory)
            re_score_mask = tf.sequence_mask(re_seq_length, maxlen=tf.shape(retrieval_memory)[1])

            def retrieval_attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(re_v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)
            retrieval_context, _ = retrieval_attention(word_output_state_, re_memory_keys, retrieval_memory, re_score_mask)
            # retrieval_context = retrieval_context * re_gate(tf.concat([word_output_state_, retrieval_context], axis=1))

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            # inp = tf.concat([x, mt, retrieval_context], axis=1)
            inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder_word(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, re_utterance_cnt, output_layer,
                                 dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_word_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("sen_attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="sen_memory_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="sen_query_layer")
            re_word_v = tf.get_variable("wor_attention_v", [hidden_dim])
            re_memory_word_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="word_memory_layer")
            re_query_word_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="word_query_layer")

            # retrieval attention
            re_sen_num = tf.shape(retrieval_memory)[1]
            re_word_num = tf.shape(retrieval_memory)[2]
            re_hidden_dim = retrieval_memory.shape[3]
            re_sen_mask = tf.sequence_mask(re_utterance_cnt, maxlen=re_sen_num)  # b, sn
            re_word_mask = tf.sequence_mask(tf.reshape(re_seq_length, [batch_size * re_sen_num]), maxlen=re_word_num)  # b * sn, wn
            re_word_mask = tf.reshape(re_word_mask, [batch_size, re_sen_num * re_word_num])  # b, sn * wn
            # mask memory and get keys
            retrieval_memory = tf.reshape(retrieval_memory, [batch_size, re_sen_num * re_word_num, re_hidden_dim])  # b, sn * wn, h
            retrieval_memory = tf.cast(tf.expand_dims(re_word_mask, -1), dtype=tf.float32) * retrieval_memory
            re_memory_keys = re_memory_word_linear(retrieval_memory)  # b, sn * wn, h

            re_gate = tf.layers.Dense(units=re_hidden_dim, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            def retrieval_attention(query, keys, values, mask_word, mask_sen):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    # word weight
                    # 1. wy
                    y = re_query_word_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # 2. v * tanh(wx + wy)
                    s = tf.reduce_sum(re_word_v * tf.tanh(keys + y), axis=2)
                    # 3. mask and normalize
                    s = _maybe_mask_score(s, mask_word)
                    s = tf.reshape(s, [-1, re_sen_num, re_word_num])
                    aa = tf.nn.softmax(s, axis=2)
                    # sentence context
                    sen_con = tf.reshape(aa, [batch_size, re_sen_num * re_word_num, 1]) * values
                    sen_con = tf.reshape(sen_con, [batch_size, re_sen_num, re_word_num, re_hidden_dim])
                    sen_con = tf.reduce_sum(sen_con, axis=2)

                    # sentence weight
                    # 1. wx
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    sen_con_key = re_memory_linear(sen_con)
                    # 2. v * tanh(wx + wy)
                    s = tf.reduce_sum(re_v * tf.tanh(sen_con_key + y), axis=2)
                    # 3. mask and normalize
                    s = _maybe_mask_score(s, mask_sen)
                    aa = tf.nn.softmax(s, axis=1)
                    # multi sen context
                    mul_sen = tf.reduce_sum(tf.reshape(aa, [batch_size, re_sen_num, 1]) * sen_con, axis=1)

                return mul_sen, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)
            retrieval_context, _ = retrieval_attention(word_output_state_, re_memory_keys, retrieval_memory, re_word_mask, re_sen_mask)
            retrieval_context = retrieval_context * re_gate(tf.concat([word_output_state_, retrieval_context], axis=1))

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, mt, retrieval_context], axis=1)
            # inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder_word_one(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, re_utterance_cnt, output_layer,
                                 dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_word_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("sen_attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="sen_memory_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="sen_query_layer")

            # retrieval attention
            re_sen_num = tf.shape(retrieval_memory)[1]
            re_word_num = tf.shape(retrieval_memory)[2]
            re_hidden_dim = retrieval_memory.shape[3]
            re_sen_mask = tf.sequence_mask(re_utterance_cnt, maxlen=re_sen_num)  # b, sn
            re_word_mask = tf.sequence_mask(tf.reshape(re_seq_length, [batch_size * re_sen_num]), maxlen=re_word_num)  # b * sn, wn
            re_word_mask = tf.reshape(re_word_mask, [batch_size, re_sen_num * re_word_num])  # b, sn * wn
            # mask memory and get keys
            retrieval_memory = tf.reshape(retrieval_memory, [batch_size, re_sen_num * re_word_num, re_hidden_dim])  # b, sn * wn, h
            retrieval_memory = tf.cast(tf.expand_dims(re_word_mask, -1), dtype=tf.float32) * retrieval_memory
            re_memory_keys = re_memory_linear(retrieval_memory)  # b, sn * wn, h

            re_gate = tf.layers.Dense(units=re_hidden_dim, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            def retrieval_attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(re_v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])

                return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)
            retrieval_context, _ = retrieval_attention(word_output_state_, re_memory_keys, retrieval_memory, re_word_mask)
            retrieval_context = retrieval_context * re_gate(tf.concat([word_output_state_, retrieval_context], axis=1))

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, mt, retrieval_context], axis=1)
            # inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder_word_one_add(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, re_utterance_cnt, output_layer,
                                 dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_word_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("sen_attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="sen_memory_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="sen_query_layer")

            # retrieval attention
            re_sen_num = tf.shape(retrieval_memory)[1]
            re_word_num = tf.shape(retrieval_memory)[2]
            re_hidden_dim = retrieval_memory.shape[3]
            re_sen_mask = tf.sequence_mask(re_utterance_cnt, maxlen=re_sen_num)  # b, sn
            re_word_mask = tf.sequence_mask(tf.reshape(re_seq_length, [batch_size * re_sen_num]), maxlen=re_word_num)  # b * sn, wn
            re_word_mask = tf.reshape(re_word_mask, [batch_size, re_sen_num * re_word_num])  # b, sn * wn
            # mask memory and get keys
            retrieval_memory = tf.reshape(retrieval_memory, [batch_size, re_sen_num * re_word_num, re_hidden_dim])  # b, sn * wn, h
            retrieval_memory = tf.cast(tf.expand_dims(re_word_mask, -1), dtype=tf.float32) * retrieval_memory
            re_memory_keys = re_memory_linear(retrieval_memory)  # b, sn * wn, h

            re_gate = tf.layers.Dense(units=re_hidden_dim, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            re_linear = tf.layers.Dense(units=hidden_dim * 2, activation=None, use_bias=None, name="value_layer")

            def retrieval_attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(re_v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])

                return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)
            retrieval_context, _ = retrieval_attention(word_output_state_, re_memory_keys, retrieval_memory, re_word_mask)
            gated = re_gate(tf.concat([word_output_state_, retrieval_context], axis=1))
            retrieval_context = retrieval_context * gated
            retrieval_context = re_linear(retrieval_context)

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
            else:
                x = dec_inputs[:, w_step, :]
            # inp = tf.concat([x, mt, retrieval_context], axis=1)
            inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits)
        )

        sen_outputs, sen_logits = word_results[-2], word_results[-1]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits


def apart_attention_decoder_word_one_copy(cell, initial_state, memory, memory_sequence_length, retrieval_memory, re_seq_length, re_utterance_cnt, output_layer,
                                 dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps

    with tf.variable_scope("apart_attention_decoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("topic_attention", reuse=tf.AUTO_REUSE):
            v = tf.get_variable("attention_v", [hidden_dim])
            memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
            query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

            memory = _mask_memory(memory, memory_sequence_length)
            meory_keys = memory_linear(memory)
            mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

            def attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Bahdanau_Attention"):
                    y = query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])
                return d, aa

        with tf.variable_scope("retrieval_word_attention", reuse=tf.AUTO_REUSE):
            re_v = tf.get_variable("sen_attention_v", [hidden_dim])
            re_memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="sen_memory_layer")
            re_query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="sen_query_layer")

            # retrieval attention
            re_sen_num = tf.shape(retrieval_memory)[1]
            re_word_num = tf.shape(retrieval_memory)[2]
            re_hidden_dim = retrieval_memory.shape[3]
            re_sen_mask = tf.sequence_mask(re_utterance_cnt, maxlen=re_sen_num)  # b, sn
            re_word_mask = tf.sequence_mask(tf.reshape(re_seq_length, [batch_size * re_sen_num]), maxlen=re_word_num)  # b * sn, wn
            re_word_mask = tf.reshape(re_word_mask, [batch_size, re_sen_num * re_word_num])  # b, sn * wn
            # mask memory and get keys
            retrieval_memory = tf.reshape(retrieval_memory, [batch_size, re_sen_num * re_word_num, re_hidden_dim])  # b, sn * wn, h
            retrieval_memory = tf.cast(tf.expand_dims(re_word_mask, -1), dtype=tf.float32) * retrieval_memory
            re_memory_keys = re_memory_linear(retrieval_memory)  # b, sn * wn, h

            re_gate = tf.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True, name="gate_layer")

            def retrieval_attention(query, keys, values, mask):
                """Put attention masks on hidden using hidden_features and query."""
                with tf.variable_scope("Retrival_Bahdanau_Attention"):
                    y = re_query_linear(query)
                    y = tf.reshape(y, [-1, 1, hidden_dim])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(re_v * tf.tanh(keys + y), axis=2)
                    s = _maybe_mask_score(s, mask)
                    aa = tf.nn.softmax(s)

                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                    d = tf.reshape(d, [batch_size, -1])

                return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])
        last_pgen = tf.zeros([batch_size, 1], dtype=tf.float32)
        last_re_dists = tf.zeros_like(re_word_mask, dtype=tf.float32)

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        p_gens = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        re_dists = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_, p_gens_, re_dists_, last_pgen_, last_re_dists_):

            mt, _ = attention(word_output_state_, meory_keys, memory, mem_score_mask)

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step, last_pgen_, last_re_dists_)
                x.set_shape([batch_size, 300])
            else:
                x = dec_inputs[:, w_step, :]
            inp = tf.concat([x, mt], axis=1)
            # inp = tf.concat([x, mt + retrieval_context], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            # pgen
            retrieval_context, re_dist = retrieval_attention(cell_output, re_memory_keys, retrieval_memory, re_word_mask)
            gated = re_gate(tf.concat([retrieval_context, word_state_new_[1].c, word_state_new_[1].h, x], axis=1))

            p_gens_new_ = p_gens_.write(w_step, gated)
            re_dists_new_ = re_dists_.write(w_step, re_dist)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_, p_gens_new_, re_dists_new_, gated, re_dist

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits, p_gens, re_dists, last_pgen, last_re_dists)
        )

        sen_outputs, sen_logits = word_results[3], word_results[4]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

        p_gens, re_dists = word_results[5], word_results[6]
        p_gens = tf.transpose(p_gens.stack(), [1, 0, 2])
        re_dists = tf.transpose(re_dists.stack(), [1, 0, 2])

    return sen_outputs, sen_logits, p_gens, re_dists


def coverage_attention_decoder(cell, initial_state, memory, memory_sequence_length, output_layer,
                               dec_inputs, dec_inputs_length, config, loop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    emb_dim = config.emb_dim
    word_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_gen_steps
    dec_mask = tf.sequence_mask(dec_inputs_length, maxlen=word_num, dtype=tf.float32)

    num_keywords = config.num_keywords

    with tf.variable_scope("coverage_attention_decoder", reuse=tf.AUTO_REUSE):

        gate = tf.ones([batch_size, num_keywords])
        atten_sum = tf.zeros([batch_size, num_keywords])

        with tf.variable_scope("coverage"):
            u_f = tf.get_variable("u_f", [num_keywords * emb_dim, num_keywords])
            res1 = tf.sigmoid(tf.matmul(tf.reshape(memory, [batch_size, -1]), u_f))
            if loop_function is not None:
                phi_res = config.max_gen_steps * res1
            else:
                phi_res = tf.reduce_sum(dec_mask, 1, keepdims=True) * res1

        v = tf.get_variable("attention_v", [hidden_dim])
        memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
        query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

        memory = _mask_memory(memory, memory_sequence_length)
        meory_keys = memory_linear(memory)
        mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

        def attention(query, keys, values, mask, gate_):
            """Put attention masks on hidden using hidden_features and query."""
            with tf.variable_scope("Bahdanau_Attention"):
                y = query_linear(query)
                y = tf.reshape(y, [-1, 1, hidden_dim])
                # Attention mask is a softmax of v^T * tanh(...).
                s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                # gate score
                s = s * gate_
                s = _maybe_mask_score(s, mask)
                aa = tf.nn.softmax(s)

                # Now calculate the attention-weighted vector d.
                d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                d = tf.reshape(d, [batch_size, -1])

            return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])

        word_step = tf.constant(0, dtype=tf.int32)

        sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)
        sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=True)

        def inter_cond(w_step, *_):
            return tf.less(w_step, word_num)

        def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_, gate_, atten_sum_):

            mt, att_prob = attention(word_output_state_, meory_keys, memory, mem_score_mask, gate_)

            # change gate and atten_sum
            gate_new_ = gate_ - (att_prob / phi_res)

            if loop_function is not None:
                x = loop_function(word_output_state_, w_step)
                atten_sum_new_ = atten_sum_
            else:
                x = dec_inputs[:, w_step, :]
                atten_sum_new_ = atten_sum_ + att_prob * dec_mask[:, w_step: w_step + 1]
            inp = tf.concat([x, mt], axis=1)

            (cell_output, word_state_new_) = cell(inp, word_state_)
            word_output_state_new_ = cell_output
            sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

            logit = output_layer(cell_output)
            sen_logits_new_ = sen_logits_.write(w_step, logit)

            return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_, gate_new_, atten_sum_new_

        word_results = tf.while_loop(
            inter_cond,
            inter_body,
            loop_vars=(word_step, state, output_state, sen_outputs, sen_logits, gate, atten_sum)
        )

        sen_outputs, sen_logits = word_results[3], word_results[4]
        sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
        sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

    return sen_outputs, sen_logits, phi_res, atten_sum


def dynamic_hier_attention_decoder(cell, initial_state, memory, memory_sequence_length, output_layer,
                                   dec_inputs, dec_inputs_length, config, loop_function=None, stop_function=None):
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    sentence_num = tf.shape(dec_inputs)[1] if loop_function is None else config.max_utterance_cnt
    word_num = tf.shape(dec_inputs)[2] if loop_function is None else config.max_gen_steps
    clear_after_read = True if loop_function is None else False
    dynamic_size = True if loop_function is None else False

    with tf.variable_scope("hier_attention_decoder", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("attention_v", [hidden_dim])
        memory_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=False, name="memory_layer")
        query_linear = tf.layers.Dense(units=hidden_dim, activation=None, use_bias=True, name="query_layer")

        memory = _mask_memory(memory, memory_sequence_length)
        meory_keys = memory_linear(memory)
        mem_score_mask = tf.sequence_mask(memory_sequence_length, maxlen=tf.shape(memory)[1])

        def attention(query, keys, values, mask):
            """Put attention masks on hidden using hidden_features and query."""
            with tf.variable_scope("Bahdanau_Attention"):
                y = query_linear(query)
                y = tf.reshape(y, [-1, 1, hidden_dim])
                # Attention mask is a softmax of v^T * tanh(...).
                s = tf.reduce_sum(v * tf.tanh(keys + y), axis=2)
                s = _maybe_mask_score(s, mask)
                aa = tf.nn.softmax(s)

                # Now calculate the attention-weighted vector d.
                d = tf.reduce_sum(tf.reshape(aa, [batch_size, -1, 1]) * values, axis=1)
                d = tf.reshape(d, [batch_size, -1])
            return d, aa

        state = initial_state
        output_state = tf.zeros([batch_size, hidden_dim])
        last_sen_hidden = tf.zeros([batch_size, word_num, hidden_dim])
        sentence_step = tf.constant(0, dtype=tf.int32)

        outputs = tf.TensorArray(dtype=tf.float32, size=sentence_num, dynamic_size=dynamic_size, clear_after_read=clear_after_read)
        logits = tf.TensorArray(dtype=tf.float32, size=sentence_num, dynamic_size=dynamic_size, clear_after_read=clear_after_read)
        output_lens = tf.TensorArray(dtype=tf.int64, size=sentence_num, dynamic_size=dynamic_size, clear_after_read=clear_after_read)

        def loop_cond(s_step, *_):
            return tf.less(s_step, sentence_num)

        def loop_body(s_step, state_, output_state_, last_sen_, outputs_, logits_, output_lens_):
            word_step = tf.constant(0, dtype=tf.int32)

            sen_outputs = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=dynamic_size)
            sen_logits = tf.TensorArray(dtype=tf.float32, size=word_num, dynamic_size=dynamic_size)

            def update_memory():
                if loop_function is not None:
                    last_sen_length_ = stop_function(logits_.read(s_step - 1))
                else:
                    last_sen_length_ = dec_inputs_length[:, s_step - 1]

                last_sen_keys = memory_linear(last_sen_)
                last_sen_mask = tf.sequence_mask(last_sen_length_, maxlen=word_num)

                concat_key_ = tf.concat([meory_keys, last_sen_keys], axis=1)
                concat_value_ = tf.concat([memory, last_sen_], axis=1)
                score_mask_ = tf.concat([mem_score_mask, last_sen_mask], axis=1)
                return concat_key_, concat_value_, score_mask_

            def org_memory():
                return meory_keys, memory, mem_score_mask

            concat_key, concat_value, score_mask = tf.cond(s_step > 0, update_memory, org_memory)

            def inter_cond(w_step, *_):
                return tf.less(w_step, word_num)

            def inter_body(w_step, word_state_, word_output_state_, sen_outputs_, sen_logits_):

                mt, _ = attention(word_output_state_, concat_key, concat_value, score_mask)

                if loop_function is not None:
                    x = loop_function(word_output_state_, w_step)
                else:
                    x = dec_inputs[:, s_step, w_step, :]
                inp = tf.concat([x, mt], axis=1)

                (cell_output, word_state_new_) = cell(inp, word_state_)
                word_output_state_new_ = cell_output
                sen_outputs_new_ = sen_outputs_.write(w_step, cell_output)

                logit = output_layer(cell_output)
                sen_logits_new_ = sen_logits_.write(w_step, logit)

                return w_step + 1, word_state_new_, word_output_state_new_, sen_outputs_new_, sen_logits_new_

            word_results = tf.while_loop(
                inter_cond,
                inter_body,
                loop_vars=(word_step, state_, output_state_, sen_outputs, sen_logits)
            )

            state_new_ = word_results[1]
            state_new_ = (tf.nn.rnn_cell.LSTMStateTuple(tf.stop_gradient(state_new_[0].c), tf.stop_gradient(state_new_[0].h)),
                          tf.nn.rnn_cell.LSTMStateTuple(tf.stop_gradient(state_new_[1].c), tf.stop_gradient(state_new_[1].h)))

            sen_outputs, sen_logits = word_results[-2], word_results[-1]
            sen_outputs = tf.transpose(sen_outputs.stack(), [1, 0, 2])
            sen_logits = tf.transpose(sen_logits.stack(), [1, 0, 2])

            # mask hidden state
            if loop_function is not None:
                sen_lengths = stop_function(sen_logits)
            else:
                sen_lengths = dec_inputs_length[:, s_step]
            sen_outputs = _mask_memory(sen_outputs, sen_lengths)

            outputs_new_ = outputs_.write(s_step, sen_outputs)
            logits_new_ = logits_.write(s_step, sen_logits)
            output_lens_new_ = output_lens_.write(s_step, sen_lengths)

            if loop_function is not None:
                sen_outputs.set_shape([batch_size, word_num, hidden_dim])

            return s_step + 1, state_new_, word_results[2], sen_outputs, outputs_new_, logits_new_, output_lens_new_

        sentence_results = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(sentence_step, state, output_state, last_sen_hidden, outputs, logits, output_lens),
        )
        outputs, logits, output_lens = sentence_results[4], sentence_results[5], sentence_results[6]
        outputs = tf.transpose(outputs.stack(), [1, 0, 2, 3])
        logits = tf.transpose(logits.stack(), [1, 0, 2, 3])
        output_lens = tf.transpose(output_lens.stack(), [1, 0])

    return outputs, logits, output_lens


score_mask_value = tf.as_dtype(tf.float32).as_numpy_dtype(-np.inf)


def _maybe_mask_score(score, score_mask):
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def _mask_memory(memory, memory_sequence_length):
    maxlen = tf.shape(memory)[1]
    seq_mask = tf.sequence_mask(memory_sequence_length, maxlen=maxlen, dtype=tf.float32)
    seq_mask = tf.expand_dims(seq_mask, -1)
    return memory * seq_mask


def get_max_loop_function(embedding, startids, output_layer=None):
    def max_loop_function(prev, w_step):
        def get_id():
            prev_ = output_layer(prev)
            return tf.argmax(prev_, 1, output_type=tf.int32)
        prev_symbol = tf.cond(tf.equal(w_step, 0), lambda: startids, get_id)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        return emb_prev

    return max_loop_function


def get_max_and_copy_loop_function(embedding, startids, re_indices, shape, output_layer=None):
    def max_loop_function(prev, w_step, pgen, re_dist):
        def get_id():
            prev_ = tf.nn.softmax(output_layer(prev))
            # print('----------', prev_.shape)
            dist = tf.scatter_nd(indices=re_indices, updates=re_dist, shape=shape)
            # print('----------', dist.shape)
            prev_ = pgen * prev_ + (1 - pgen) * dist
            # print('----------', prev_.shape)
            return tf.argmax(prev_, 1, output_type=tf.int32)
        prev_symbol = tf.cond(tf.equal(w_step, 0), lambda: startids, get_id)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        return emb_prev

    return max_loop_function


def get_stop_function(stopid):
    def stop_function(sen_oupt_logits):
        sen_symbol = tf.argmax(sen_oupt_logits, -1)

        def fn(single_sen):
            index = tf.where(tf.equal(single_sen, stopid))
            index = tf.cond(tf.equal(tf.shape(index)[0], 0),
                            lambda: tf.cast(tf.shape(single_sen)[0], tf.int64),
                            lambda: index[0, 0] + 1)
            return index
        sen_length = tf.map_fn(fn, sen_symbol)
        return sen_length

    return stop_function


