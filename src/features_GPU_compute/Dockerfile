# Adapted from https://github.com/yjxiong/temporal-segment-networks/blob/docker/Dockerfile
# and https://github.com/AlanZhang1995/LeinaoPAI/blob/master/images/temporal-segment-networks2/Dockerfile

FROM nvidia/cuda:9.0-cudnn7-devel



RUN apt-get update
RUN apt-get -qq install -y python2.7 libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev unzip zip cmake
RUN apt-get -qq install --no-install-recommends libboost1.58-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev wget python-pip git-all libzip-dev

# RUN PYTHON INSTALL
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy scipy sklearn scikit-image

RUN apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

RUN apt-get -qq install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
RUN apt-get -qq install --no-install-recommends libboost1.58-all-dev
RUN apt-get -qq install libgflags-dev libgoogle-glog-dev liblmdb-dev

# install Dense_Flow dependencies
RUN apt-get -qq install libzip-dev

WORKDIR /app

ADD . /app

# Get code

RUN nvcc --version

ENV LIBRARY_PATH=/usr/local/cuda/lib64
# RUN bash -e build_all.sh

##################################
# MPI addition from https://github.com/AlanZhang1995/LeinaoPAI/blob/master/images/temporal-segment-networks2/Dockerfile
RUN mkdir /app/lib/MPI

WORKDIR /app/lib/MPI

RUN apt-get install -y openssh-client openssh-server && \
    pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple opencv-python && \
    wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz && \
    tar -zxvf mpich-3.2.1.tar.gz && \
	cd mpich-3.2.1 &&\
	./configure -prefix=/app/lib/MPI/mpich && \
    make && \
    make install  && \
    cd /app && \
	MPI_PREFIX=/app/lib/MPI/mpich/ bash -e build_all.sh MPI_ON

ENV PATH=/app/lib/MPI/mpich/bin:$PATH

WORKDIR /app
##################################

RUN bash scripts/get_reference_models.sh
RUN bash scripts/get_init_models.sh

RUN pip install ipython protobuf
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
RUN pip install torchvision

RUN apt-get -qq install -y vim

CMD ["ipython"]
