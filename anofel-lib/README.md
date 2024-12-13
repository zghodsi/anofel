# AnoFel-lib 
This directory contains the source code for AnoFel federated learning. The secure aggregation implementation is based on the [PPFL library](https://github.com/meandmymind/hybrid-approach-to-ppfl).

## Overview
Experiment settings can be changed in the [`config`](src/config.py) file. 

## Build Guide
1. With docker installed, run:
```
docker build -t anofel-lib .
```
2. Run test with:
```
bash rundocker.sh
python3 main.py
```
