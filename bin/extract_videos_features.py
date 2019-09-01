"""
Script forL
1. Extracting features from videos (visual features using resnet and c3d model)
2. Extracting features from videos metadata (tiltles) using fasttext
"""

import argparse
from datetime import datetime
import json
import os
import pickle
import yaml

import fasttext
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from mmsentiment.extractors.visual import ResNetExtractor, C3DExtractor
from mmsentiment.extractors.textual import FastTextExtractor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to the config file",
                        required=True)
    parser.add_argument("--extract_resnet",
                        action="store_true")
    parser.add_argument("--extract_c3d",
                        action="store_true")
    parser.add_argument("--extract_fasttext",
                        action="store_true")
    return parser.parse_args()


def save_file(data, filename_prefix, filename, output_dir):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")

    save_path = os.path.join(output_dir,
                             "{}_{}_{}.pkl".format(filename_prefix, filename, dt_string))

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print("File saved as: ", save_path)


def extract_visual_features(extractor, video_paths):

    video_features = dict()
    print("Total files found: {}".format(len(video_paths)))
    for i, video_path in enumerate(video_paths):
        print("Processing file {}. {}".format(i, video_path))
        video_id = os.path.basename(video_path).split('.')[0]
        features = extractor.extract(video_path)
        video_features[video_id] = features

    return video_features


def extract_textual_features(extractor, texts_data: dict):

    textual_features = dict()
    for id_, text in texts_data.items():
        features = extractor.extract(text)
        textual_features[id_] = features

    return textual_features


def extract_resnet(videos_paths, resnet_model_path, output_file_prefix, output_dir):
    print("Extracting resnet features")

    resnet_model = load_model(resnet_model_path)
    resnet_extractor = ResNetExtractor(model=resnet_model,
                                       preprocess_function=preprocess_input)
    resnet_features = extract_visual_features(resnet_extractor, videos_paths)
    save_file(resnet_features, output_file_prefix, 'resnet',
              output_dir)

    del resnet_model
    del resnet_features


def extract_c3d(videos_paths, c3d_model_path, output_file_prefix, output_dir):
    print("Extracting c3d features")

    c3d_model = load_model(c3d_model_path)
    c3d_extractor = C3DExtractor(model=c3d_model)
    c3d_features = extract_visual_features(c3d_extractor, videos_paths)
    save_file(c3d_features, output_file_prefix, 'c3d',
              output_dir)

    del c3d_model
    del c3d_features


def extract_fasttext_metadata(metadata_file_path, fasttext_model_path,
                              output_file_prefix, output_dir):

    print("Loading fasttext model")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)

    print("Extracting features from titles")
    titles = {id_: data['title'] for id_, data in metadata.items()}
    fasttext_extractor = FastTextExtractor(model=fasttext_model)
    fasttext_features = extract_textual_features(fasttext_extractor, titles)
    save_file(fasttext_features, output_file_prefix, 'fasttext_titles', output_dir)

    del fasttext_model
    del fasttext_features


def main():
    args = parse_arguments()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f)

    videos_paths = [os.path.join(config['videos_paths'], filename)
                    for filename in os.listdir(config['videos_paths'])]

    if args.extract_resnet:
        extract_resnet(videos_paths, config['resnet_model_path'], config['output_file_prefix'],
                       config['output_dir'])

    if args.extract_c3d:
        extract_c3d(videos_paths, config['c3d_model_path'], config['output_file_prefix'],
                    config['output_dir'])

    if args.extract_fasttext:
        extract_fasttext_metadata(config['videos_metadata_path'], config['fasttext_model_path'],
                                  config['output_file_prefix'], config['output_dir'])


if __name__ == "__main__":
    main()
