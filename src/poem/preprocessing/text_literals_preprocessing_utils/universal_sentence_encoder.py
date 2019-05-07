# -*- coding: utf-8 -*-

"""Wrapper for the universal sentence encoder."""
from src.kupp.text_preprocessing_utils.AbstractSentenceEncoder import AbstractSentenceEncoder
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


class USEncoder(AbstractSentenceEncoder):
    def __init__(self):
        # model 2 corresponds to the deep averaging network
        self.pretrained_model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def encode(self, texts: list) -> np.array:
        """Encode text passages."""

        embeddings = self.pretrained_model(texts)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embedded_sentences = np.array(sess.run(embeddings))

        return embedded_sentences
