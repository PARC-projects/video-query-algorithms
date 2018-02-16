# video-query-algorithms
Rough README for starters:

Pipeline:
1.  use build_wof_clips.py to build all the rgb and warped optical flow jpeg files, in a specified directory structure.
      This program uses Temporal Segment Networks (TSN) code.
      
2.  use calcSig_wOF_ensemble.sh to compute features for all clips, stored in csv files within a specified directory structure.
      This script calls calcSig_wOF.py, which uses Temporal Segment Networks code to compute features for CNN defined in Caffe.
      
3.  create_db.py can create the postgres "features", "cnn_streams", and "video_clips" tables, as well as load the "cnn_streams" table.  The cnn_streams table is just a lookup table, whose sole purpose is to avoid problems with possibly spelling or capitalizing stream names in different ways.

4.   load_db.py loads the "features" table, given csv files with features from steps 1 and 2.

5.  compute_similarities.py computes similarity values for the similarity between a given reference clip and all other clips.
      All of the code in this repository is a work in progress, but this file in particular is currently being coded. 
