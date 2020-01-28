############################
# Compute
############################

FROM bitxiong/tsn

# create a directory for mounting and external folder contianing the videos of interest
RUN mkdir /video_data

# copy models and executables for computing embeddings to the temporal segment networks code directory
COPY ./src/features_GPU_compute/models /app/models/
COPY ./src/features_GPU_compute/calcSig_wOF_ensemble.sh /app/scripts/
COPY ./src/features_GPU_compute/calcSig_wOF.py /app/tools/
COPY ./src/features_GPU_compute/build_wof_clips.py /app/tools/

# set environ variables
ENV PYTHONPATH=$PYTHONPATH:/app:/app/lib/caffe-action/python:/app/pyActionRecog:
ENV TSN_ROOT='/app'
ENV COMPUTE_EPS='.000003'  RANDOM_SEED='73459912436'

# script to remove any spaces from names of videos - use regular expressions
# find . -type f -name "* *.xml" -exec rename "s/\s/_/g" {} \;
# # V100-1 has 6 cpu’s
  #!bash scripts/calcSig_wOF_ensemble.sh tsn/Video_clips/RunningReferenceSet tsn/Video_features 6
  #!bash scripts/calcSig_wOF_ensemble.sh tsn/Video_clips/RunningSearchSet tsn/Video_features 6
  #!bash scripts/calcSig_wOF_ensemble.sh tsn/Video_clips/UdogYogaReferenceSet tsn/Video_features 6
  #!bash scripts/calcSig_wOF_ensemble.sh tsn/Video_clips/UdogYogaSearchSet tsn/Video_features 6

CMD ["/bin/bash"]
