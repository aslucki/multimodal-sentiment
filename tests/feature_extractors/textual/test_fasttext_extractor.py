import os

import fasttext

from mmsentiment.extractors.textual import FastTextExtractor

MODEL = fasttext.load_model(os.path.join('.', 'models', 'crawl-300d-2M-subword.bin'))


def test_extract_fasttext_features():
    exctractor = FastTextExtractor(MODEL)
    sentence = "This is a sample sentence"

    features = exctractor.extract(sentence)

    assert  features.shape == (300,)