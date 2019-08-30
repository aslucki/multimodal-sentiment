from collections import deque, namedtuple

import av
import numpy as np
from PIL import Image


from .extractor import Extractor


class VisualExtractor(Extractor):

    def __init__(self, model, method='keyframes', image_size=(224, 224),
                 preprocess_function=None):
        self._model = model
        self._method = method
        self._image_size = image_size
        self._preprocess_function = preprocess_function

    def extract(self, file_path):
        extractor = self._get_extractor()

        return extractor(file_path)

    def _get_extractor(self):
        if self._method == 'keyframes':
            return self._extract_keyframes_features
        elif self._method == 'C3D':
            return self._extract_c3d_features

    def _extract_keyframes_features(self, file_path):
        """
        Extracts visual features from video keyframes using
        a specified model.

        :param file_path:
        :return: Averaged feature vector for a whole video
        """
        frames = self._process_video(file_path, target_size=self._image_size)
        try:
            if self._preprocess_function:
                processed = self._preprocess_function(frames.keyframes)
            else:
                processed = frames.keyframes
            features = self._model.predict(processed)
            features = np.mean(features, axis=0)

            return np.squeeze(features)

        except NameError as e:
            print(e)

    def _extract_c3d_features(self, file_path):
        frames = self._process_video(file_path,
                                     target_size=self._image_size).frames

        features = []
        video_chunks = self._chunks(frames)
        while True:
            try:
                video_chunk = next(video_chunks)
                video_chunk = np.expand_dims(video_chunk,
                                             axis=0)
                c3d_features = self._model.predict(video_chunk)
                features.append(c3d_features)
            except StopIteration:
                break

        features = np.asarray(features)
        features = np.mean(features, axis=0)
        normalized = features / np.linalg.norm(features)

        return np.squeeze(normalized)

    @staticmethod
    def _process_video(file_path, target_size=(224, 224)):
        container = av.open(file_path)
        stream = container.streams.video[0]

        video_data = namedtuple('video_data', 'frames keyframes')
        all_frames = []
        keyframes = []
        for frame in container.decode(stream):

            if frame.key_frame:
                img = frame.to_image()
                img = img.resize(target_size, Image.ANTIALIAS)
                keyframes.append(np.array(img,
                                          dtype=np.float32))
            img = frame.to_image()
            img = img.resize(target_size, Image.ANTIALIAS)
            all_frames.append(np.array(img,
                                       dtype=np.uint8))
        return video_data(frames=np.asarray(all_frames),
                          keyframes=np.asarray(keyframes))

    @staticmethod
    def _chunks(iterable, chunk_size=16, overlap=8):

        queue = deque(maxlen=chunk_size)
        it = iter(iterable)
        i = 0

        for i in range(chunk_size):
            queue.append(next(it))
        while True:
            yield np.array(queue)
            for i in range(chunk_size - overlap):
                queue.append(next(it))
