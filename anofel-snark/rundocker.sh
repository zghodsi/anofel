#! /bin/bash

docker rm -f anofel-snark-test 2>/dev/null || true

docker run -it \
    --name anofel-snark-test \
    -w /anofel-snark/build/src/test \
    anofel-snark

