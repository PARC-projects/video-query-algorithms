#!/usr/bin/env bash
# Preconditions:
# Set TSN_ROOT environment variable to the root of the temporal segment networks code
# Set TSN_ENVIRONMENT to the name of the conda environment used to run temporal segment netowrks code,
#      or setup the virtual environment beforehand
source activate $TSN_ENVIRON

SRC_FOLDER=$1   # folder with rgb frames and warped optical flow images of clips, SRC_FOLDER/{video name}/{clip folders}/*.jpg
OUT_FOLDER=$2   # folder for the computed features
NUM_WORKER=$3   # number of processes, e.g. 24.  If greater than number of gpu's, will load multiple on each gpu

# features computed using split1 models
python $TSN_ROOT/tools/calcSig_wOF.py ${SRC_FOLDER} \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_rgb_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split1_tsn_rgb_bn_inception_wOF_test3_iter_3500.caffemodel \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_flow_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split1_tsn_flow_bn_inception_wOF_test3_iter_18000.caffemodel \
 --outFeatures_dir ${OUT_FOLDER}  \
 --num_worker ${NUM_WORKER}  --modelname UCF101_split1 --gpus 0 1 2 3 4 5 6 7

# features computed using split2 models
python $TSN_ROOT/tools/calcSig_wOF.py ${SRC_FOLDER} \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_rgb_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split2_tsn_rgb_bn_inception_wOF_test3_iter_3500.caffemodel \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_flow_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split2_tsn_flow_bn_inception_wOF_test3_iter_18000.caffemodel \
 --outFeatures_dir ${OUT_FOLDER}  \
 --num_worker ${NUM_WORKER}  --modelname UCF101_split2 --gpus 0 1 2 3 4 5 6 7

# features computed using split3 models
 python $TSN_ROOT/tools/calcSig_wOF.py ${SRC_FOLDER} \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_rgb_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split3_tsn_rgb_bn_inception_wOF_test3_iter_3500.caffemodel \
 $TSN_ROOT/models/ucf101/tsn_bn_inception_flow_deploy.prototxt \
 $TSN_ROOT/models/ucf101_split3_tsn_flow_bn_inception_wOF_test3_iter_18000.caffemodel \
 --outFeatures_dir ${OUT_FOLDER}  \
 --num_worker ${NUM_WORKER}  --modelname UCF101_split3 --gpus 0 1 2 3 4 5 6 7

# List of features we are considering:
#
# global_pool (default in wcalSig_wOF.py)
# inception_5a/output
# inception_4e/output
# inception_4d/output
# inception_4c/output
# inception_4b/output
# inception_4a/output
# inception_3c/output
# inception_3b/output
# inception_3a/output
#
# Choose by specifying --featureBlob and --featureBlob_size
