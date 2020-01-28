#!/bin/sh

# set environment variables
source set_environ.sh

# run broker
python broker.py

exec "$@"
