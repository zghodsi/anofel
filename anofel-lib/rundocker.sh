#! /bin/bash

NAME=0
GPU="device=$NAME"

docker rm -f anofel-lib-test-$NAME 2>/dev/null || true

docker run -it \
    --name anofel-lib-test-$NAME \
    --gpus "$GPU" \
    -v $(pwd)/data:/anofel-lib/src/data \
    -v $(pwd)/accexps:/anofel-lib/src/accexps \
    -v $(pwd)/benchmarks:/anofel-lib/src/benchmarks \
    anofel-lib:latest

