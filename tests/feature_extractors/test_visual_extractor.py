import os

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from mmsentiment.feature_extractors import VisualExtractor

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'sample_data', 'video.mp4')
RESNET_MODEL = load_model(os.path.join('.', 'models', 'resnet50_pool5.h5'))
C3D_MODEL = load_model(os.path.join('.', 'models', 'C3D_fc6.h5'))


def test_extract_keyframes_features():
    extractor = VisualExtractor(model=RESNET_MODEL,
                                method='keyframes',
                                image_size=(224, 224),
                                preprocess_function=preprocess_input)

    features = extractor.extract(VIDEO_PATH)

    assert features.shape == (2048,)


def test_extract_c3d_features():
    extractor = VisualExtractor(model=C3D_MODEL,
                                method='C3D',
                                image_size=(112, 112))

    features = extractor.extract(VIDEO_PATH)

    assert features.shape == (4096,)

