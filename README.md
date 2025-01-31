<h1 align="center">AnoFel</h1>

___AnoFel___ is a Python and C++ library for anonymous and privacy-preserving federated learning. This library was developed for the paper [AnoFel](https://arxiv.org/abs/2306.06825). 

## Overview
AnoFel library leverages several cryptographic primitives, the concept of anonymity sets, and differential privacy to support anonymous user registration and confidential model update submission during federated learning.

## Directory Structure
This repository contains the following high-level structure:
* [anofel-lib](anofel-lib) python scripts for AnoFel federated learning which incorporates secure aggregation and differential privacy. The scripts can be used to obtain runtime and accuracy benchmarks.
* [anofel-snark](anofel-snark) C++ code for AnoFel zk-SNARK for client setup and training to obtain prover and verifier runtime.


Each directory contains a Dockerfile and build guide for running experiments. Access to a GPU is recommended for running training experiments in anofel-lib.
