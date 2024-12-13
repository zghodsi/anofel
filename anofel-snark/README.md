# AnoFel-snark
This directory contains the source code for AnoFel zk-SNARK for client setup and training. The implementation is based on the [libsnark](https://github.com/scipr-lab/libsnark) library, and JubJub elliptic curve [gadgets](https://github.com/HarryR/ethsnarks/tree/master/src/jubjub).

## Overview
The source files [`test_anofel_setup.cpp`](src/test/test_anofel_setup.cpp) and [`test_anofel_training.cpp`](src/test/test_anofel_setup.cpp) build the client setup and training circuits where `tree_depth` indicates $\log(N)$ for $N$ number of clients.

## Build Guide
1. With docker installed, run:
```
docker build . -t anofel-snark
```
Alternatively, to use ccache build with the script provided
```
bash buildcontainer.sh
```
2. Run container and tests:
```
bash rundocker.sh
./test_anofel_setup
./test_anofel_training
```
