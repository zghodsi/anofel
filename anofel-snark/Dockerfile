FROM debian:11.10

RUN apt-get update && \
    apt-get install -y \
    wget unzip curl ccache \
    build-essential cmake git libgmp3-dev libprocps-dev libboost-all-dev libssl-dev pkg-config pip

RUN pip install markdown

RUN mkdir /anofel-snark
WORKDIR /anofel-snark
RUN git init
RUN git config --global url.https://github.com:/.insteadOf git://github.com/
RUN mkdir depends && cd depends && \
    echo "add_subdirectory(libsnark)" > CMakeLists.txt && \
	git submodule add https://github.com/scipr-lab/libsnark.git libsnark && \
	git submodule add https://github.com/harryr/SHA3IUF.git SHA3IUF
COPY depends/CMakeLists.txt depends/CMakeLists.txt

COPY CMakeLists.txt CMakeLists.txt

RUN git submodule update --init --recursive

COPY src/ src/

RUN mkdir build && cd build && cmake .. && CCACHE_SECONDARY_STORAGE=redis://localhost make CC=ccache -j $(nproc --ignore=1)

WORKDIR build/src
