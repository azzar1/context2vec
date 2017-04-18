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
#
# Authors: Andrea Azzarone <azzaronea@gmail.com>
#

import tensorflow as tf
import tensorflow_fold.public.blocks as td

def __build_lstm_cell(num_units, forget_bias, layer_norm, norm_gain, norm_shift, dropout_keep_prob, dropout_prob_seed, name_or_scope):
    lstm_cell = td.ScopedLayer(
        tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units,
                                                  forget_bias=forget_bias,
                                                  layer_norm=layer_norm,
                                                  norm_gain=norm_gain,
                                                  norm_shift=norm_shift,
                                                  dropout_keep_prob=dropout_keep_prob,
                                                  dropout_prob_seed=dropout_prob_seed),
            input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob),
        name_or_scope=name_or_scope)
    return lstm_cell


def build_bi_lstm(num_units, forget_bias, layer_norm, norm_gain, norm_shift, dropout_keep_prob, dropout_prob_seed):
    fw_lstm_cell=__build_lstm_cell(
        num_units = num_units, forget_bias = forget_bias,
        layer_norm = layer_norm, norm_gain = norm_gain, norm_shift = norm_shift,
        dropout_keep_prob = dropout_keep_prob, dropout_prob_seed = dropout_prob_seed,
        name_or_scope = "fw_lstm_cell")
    fw_pass=td.RNN(fw_lstm_cell)

    reverse_sequence=td.Slice(step=-1)
    bw_lstm_cell = __build_lstm_cell(
        num_units=num_units, forget_bias=forget_bias,
        layer_norm=layer_norm, norm_gain=norm_gain, norm_shift=norm_shift,
        dropout_keep_prob=dropout_keep_prob, dropout_prob_seed=dropout_prob_seed,
        name_or_scope="bw_lstm_cell")
    bw_pass = (reverse_sequence >>
               td.RNN(bw_lstm_cell))

    return td.AllOf(fw_pass, bw_pass)
