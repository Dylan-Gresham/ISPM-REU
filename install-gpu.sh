#!/bin/bash

INVENV=$(python -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')
NVCC_PATH=$(which nvcc)

if [ "$INVENV" == 1 ] && [ "$NVCC_PATH" == "" ]
then
  FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_CUDA=on" CUDACXX=$NVCC_PATH pip install --requirements requirements.txt
else
  source .env/bin/activate
  FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_CUDA=on" CUDACXX=$NVCC_PATH pip install --requirements requirements.txt pip install --requirements requirements.txt
fi