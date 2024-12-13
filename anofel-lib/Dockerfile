FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip \
    libgmp-dev libmpfr-dev libmpc-dev vim

RUN mkdir /anofel-lib
WORKDIR /anofel-lib
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src/ src/
WORKDIR src



