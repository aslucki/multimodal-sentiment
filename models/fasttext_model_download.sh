#!/bin/bash
FASTEXT_MODEL_URL=https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
FILENAME=crawl-300d-2M-subword

function validate_url(){
  if [[ `wget -q -o /dev/null -S --spider $1  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then echo "true"; fi
}

if [ `validate_url $FASTEXT_MODEL_URL > /dev/null` ]  & [ ! -f "$FILENAME.bin" ]
then
  wget $FASTEXT_MODEL_URL
  unzip "$FILENAME.zip"
  rm "$FILENAME.zip" "$FILENAME.vec"
else
  echo "Model is not available or already downloaded."
fi
