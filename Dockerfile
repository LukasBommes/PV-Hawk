FROM tensorflow/tensorflow:1.15.2-gpu-py3

WORKDIR /

# OpenCV & matplotlib dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt


##############################################################################
#
#   Pangolin Viewer
#
##############################################################################

# Install pangoling dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        pkg-config \
        libgl1-mesa-dev \
        libglew-dev \
        cmake \
        libpython2.7-dev \
        libegl1-mesa-dev \
        libwayland-dev \
        libxkbcommon-dev \
        wayland-protocols \
        libeigen3-dev \
        doxygen && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pyopengl Pillow pybind11

WORKDIR /home
RUN git clone https://github.com/lukasbommes-forked-projects/Pangolin.git pangolin

WORKDIR /home/pangolin
RUN git submodule init && git submodule update

WORKDIR /home/pangolin/build
RUN cmake .. && \
    cmake --build . && \
    cmake --build . --target doc

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


###############################################################################
#
#          Install OpenSfM dependencies
#
###############################################################################

ARG DEBIAN_FRONTEND=noninteractive

# Install apt-getable dependencies
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        libeigen3-dev \
        libgoogle-glog-dev \
        libopencv-dev \
        libsuitesparse-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install Ceres 2
RUN \
    mkdir -p /source && cd /source && \
    curl -L http://ceres-solver.org/ceres-solver-2.0.0.tar.gz | tar xz && \
    cd /source/ceres-solver-2.0.0 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF && \
    make -j4 install && \
    cd / && rm -rf /source/ceres-solver-2.0.0


##############################################################################
#
#   pyg2o hyper graph optimizer
#
##############################################################################

RUN apt-get update && \
  apt-get install -y \
    cmake \
    git \
    build-essential \
    libeigen3-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code
RUN git clone https://github.com/lukasbommes-forked-projects/g2opy.git

WORKDIR /code/g2opy/build
  RUN cmake .. \
  && make -j12 \
  && make install -j12 \
  && ldconfig

WORKDIR /code/g2opy/
RUN python setup.py install

WORKDIR /code/g2opy


###############################################################################
#
#                          Container Startup
#
###############################################################################

# Run entrypoint script
WORKDIR /pvextractor
COPY docker-entrypoint.sh /pvextractor
RUN chmod +x /pvextractor/docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
