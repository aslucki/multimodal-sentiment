import os

from tensorflow.keras.models import load_model

from mmsentiment.extractors.visual import C3DExtractor

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'sample_data', 'video.mp4')
C3D_MODEL = load_model(os.path.join('.', 'models', 'C3D_fc6.h5'))


def test_extract_c3d_features():
    extractor = C3DExtractor(model=C3D_MODEL,
                             image_size=(112, 112))

    features = extractor.extract(VIDEO_PATH)

    assert features.shape == (4096,)
