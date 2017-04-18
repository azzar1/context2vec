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


class TestSequenceTranscoder(unittest.TestCase):

    def test_sequence_length(self):
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        self.assertEqual(len(st.eval(["hello", "world"])), 2)
        self.assertEqual(len(st.eval(["hello", "world", "ciao", "mondo"])), 4)
        self.assertEqual(len(st.eval(["hello", "world", "unknown_word"])), 3)

    def test_word_embedding_size(self):
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        sequence = st.eval(["hello", "world"])
        self.assertEqual([len(w) for w in sequence], [3,3])
        
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 5)
        sequence = st.eval(["hello", "world"])
        self.assertEqual([len(w) for w in sequence], [5,5])

    def test_same_word_embedding(self):
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        sequence1 = st.eval(["hello", "world", "hello"])
        sequence2 = st.eval(["hello", "world", "hello"])
        self.assertTrue(np.array_equal(sequence1[0], sequence1[2]))
        self.assertTrue(np.array_equal(sequence1, sequence2))

    def test_unknown_word(self):
        st = sequence_transcoder.build_sequence_transcoder('src/test_data/vocab.txt', 3)
        sequence1 = st.eval(["unknown_word1", "world", "unknown_word2"])
        sequence2 = st.eval(["unknown_word3", "world", "unknown_word4"])
        self.assertTrue(np.array_equal(sequence1[0], sequence1[2]))
        self.assertTrue(np.array_equal(sequence1, sequence2))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    unittest.main()
