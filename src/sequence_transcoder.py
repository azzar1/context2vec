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

import codecs
import os

import tensorflow as tf
import tensorflow_fold.public.blocks as td


def load_vocab(filepath):
    assert os.path.exists(filepath), "%s does not exists." % filepath

    with codecs.open(filepath, encoding='utf-8') as f:
        lines = f.read().splitlines()
        assert len(
            lines) > 0, "Invald vocabulary file: %s has invalid lenght." % filepath

        vocab = dict()
        for i, word in enumerate(lines[1:]):
            vocab[word] = i
        return vocab


def word2index(vocab, word):
    if word in vocab:
        return vocab[word]
    else:
        return len(vocab)


def build_sequence_transcoder(vocab_filepath, word_embedding_size):
    vocab_size = 5

    # From words to list of integers
    vocab = load_vocab(vocab_filepath)
    words2integers = td.InputTransform(
        lambda s: [word2index(vocab, w) for w in s])

    # From interger to word embedding
    word2embedding = td.Scalar('int32') >> td.Function(
        td.Embedding(vocab_size, word_embedding_size))

    # From word to array of embeddings
    sequence_transcoder = words2integers >> td.Map(word2embedding)
    return sequence_transcoder
