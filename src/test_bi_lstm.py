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

import numpy as np
import tensorflow as tf
import tensorflow_fold.public.blocks as td
import unittest

import sequence_transcoder
import bi_lstm


class TestBiLstm(unittest.TestCase):

    def test_outputs(self):
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        bilstm = bi_lstm.build_bi_lstm(num_units=5, forget_bias=1.0,
            layer_norm=1, norm_gain=1.0, norm_shift=0.0,
            dropout_keep_prob=1.0, dropout_prob_seed=None)
        block = (st >> bilstm)

        # One output for the fw pass, one for the bw
        output = block.eval(["hello", "world", "unknown_word"])
        self.assertEqual(len(output), 2)

        # Two outputs for each pass ([h1,h2,h3] and [c3,h3])
        self.assertEqual(len(output[0]), 2)     
        self.assertEqual(len(output[1]), 2)

        #
        # Forward pass
        #
        fw_hs = output[0][0]
        fw_last_h = output[0][1][1]

        # Three as the words
        self.assertEqual(len(fw_hs), 3)
        self.assertTrue(np.array_equal(fw_hs[2], fw_last_h))

        # Five as the num_units
        self.assertTrue(len(fw_last_h), 5)  

        #
        # Backward pass
        #
        bw_hs = output[1][0]
        bw_last_h = output[1][1][1]

        # Three as the words
        self.assertEqual(len(bw_hs), 3)
        self.assertTrue(np.array_equal(bw_hs[2], bw_last_h))

        # Five as the num_units
        self.assertTrue(len(bw_last_h), 5)

        # Make sure fw and bw are not equal, it can happen but
        # it should be pretty rare.
        self.assertFalse(np.array_equal(fw_hs, bw_hs))


    # Not a real unittest, just a way to test that td.Slice(step=-1)
    # reversts a sequences
    def test_reversed(self):
        reverse_sequence=td.Slice(step=-1)
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        block = (st >> reverse_sequence)
        s1 = st.eval(["hello", "world", "ciao", "unknown_word"])
        s2 = block.eval(["hello", "world","ciao", "unknown_word"])
        self.assertTrue(np.array_equal(s1[::-1], s2))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    unittest.main()
