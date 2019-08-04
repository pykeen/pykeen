# -*- coding: utf-8 -*-

"""Wrapper for the universal sentence encoder."""

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .sentence_encoder import SentenceEncoder

__all__ = [
    'USEncoder',
]


class USEncoder(SentenceEncoder):
    """Encodes sentences using the google universal sentence encoder."""

    def __init__(self):
        # model 2 corresponds to the deep averaging network
        self.pretrained_model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def encode(self, texts: List[str]) -> np.array:
        """Encode text passages."""
        embeddings = self.pretrained_model(texts)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embedded_sentences = np.array(sess.run(embeddings))

        return embedded_sentences
