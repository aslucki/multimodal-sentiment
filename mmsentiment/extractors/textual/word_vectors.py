import re

import numpy as np

from mmsentiment.extractors.base_extractor import Extractor


class FastTextExtractor(Extractor):

    def __init__(self, model):
        self._model = model

    def extract(self, sentence):
        return self._sentence_to_vec(sentence)

    def _word_to_vec(self, word):
        try:
            return self._model[word]
        except KeyError:
            dim = self._model['cat'].shape
            return np.zeros(shape=dim)

    def _sentence_to_vec(self, sentence):
        cleaned_sentence = re.sub('\W+', '', sentence)
        words = cleaned_sentence.split()

        word_vectors = []
        for word in words:
            word_vector = self._word_to_vec(word)
            word_vectors.append(word_vector)

        word_vectors = np.asarray(word_vectors)

        return np.mean(word_vectors, axis=0)
