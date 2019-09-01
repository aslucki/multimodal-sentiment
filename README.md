# Multimodal sentiment analysis
Understanding YouTube video comments based on contextual information extracted from videos themselves.

# Usage
## Environment setup
0. Install Git LFS if necessary: https://git-lfs.github.com/
1. Clone the repository
2. Prepare environment by building the docker image:
   `make build`
3. Unpack video files (in data/raw)
4. Download fasttext model (./models.fastext_model_download.sh)

## Extract features from videos and videos metadata
0. Adjust config file if necessary (bin/config.yaml)
1. Start a container:
`make dev`
2. Install mmsentiment module:
`pip3 install -e .`
3. Execute the python script:
```
python3 bin/extract_videos_features.py \
    --config bin/config.yaml \
    --extract_resnet \
    --extract_c3d \
    --extract_fasttext
```

Features will be saved in the directory specified in config (the default is: data/processed)

### Features files format
Features are dictionaries serialized using pickle. The structure is as follows:

```
{
    '[video_id: str]': [features vector: numpy array],
    ...
}
```

### Features types
Currently we extra


