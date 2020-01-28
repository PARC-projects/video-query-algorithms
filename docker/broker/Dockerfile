############################
# Broker
############################

# pull official base image
FROM python:3.7.4-alpine

# create working directory
RUN mkdir /code/
COPY ./src/ /code/
WORKDIR /code/

# install dependencies
RUN apk add libffi-dev gcc gfortran wget openblas-dev build-base libpng-dev
# libev4 libev-dev
RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
# RUN apk --no-cache --update-cache add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip install cython
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt
COPY ./set_environ.sh /code/

# run entrypoint.sh to set environment variables and start broker
ENTRYPOINT ["/code/entrypoint.sh"]

