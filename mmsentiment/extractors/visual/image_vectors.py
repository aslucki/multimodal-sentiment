import numpy as np

from .visual_extractor import VisualExtractor


class ResNetExtractor(VisualExtractor):

    def __init__(self, model, image_size=(224, 224), preprocess_function=None):
        self._model = model
        self._image_size = image_size
        self._preprocess_function = preprocess_function

    def extract(self, file_path):
        keyframes = self.video_to_frames(file_path,
                                         image_size=self._image_size).keyframes

        features = self._extract_image_features(keyframes)

        return features

    def _extract_image_features(self, images):

        try:
            if self._preprocess_function:
                processed = self._preprocess_function(images)
            else:
                processed = images
            features = self._model.predict(processed)
            features = np.mean(features, axis=0)

            return np.squeeze(features)

        except NameError as e:
            print("Operation is not supported", e)
