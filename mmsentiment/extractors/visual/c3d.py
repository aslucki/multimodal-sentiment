from collections import deque

import numpy as np

from .visual_extractor import VisualExtractor


class C3DExtractor(VisualExtractor):

    def __init__(self, model, image_size=(112, 112)):
        self._model = model
        self._image_size = image_size

    def extract(self, file_path):
        frames = self.video_to_frames(file_path,
                                      image_size=self._image_size).frames

        features = self._extract_c3d_features(frames)

        return features

    def _extract_c3d_features(self, video_frames):

        features = []
        video_chunks = self._chunks(video_frames)
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
