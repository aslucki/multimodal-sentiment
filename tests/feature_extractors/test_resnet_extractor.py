import os

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from mmsentiment.extractors.visual import ResNetExtractor

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'sample_data', 'video.mp4')
RESNET_MODEL = load_model(os.path.join('.', 'models', 'resnet50_pool5.h5'))


def test_extract_keyframes_features():
    extractor = ResNetExtractor(model=RESNET_MODEL,
                                image_size=(224, 224),
                                preprocess_function=preprocess_input)

    features = extractor.extract(VIDEO_PATH)

    assert features.shape == (2048,)
