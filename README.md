# video-query-algorithms

### Pipeline overview:
1.  use build_wof_clips.py to build all the rgb and warped optical flow jpeg files, in a specified directory structure.
      This program calls Temporal Segment Networks (TSN) code, and it assumes the command below is being run from the main TSN directory.
      Code is written to run on a GPU compute server.

2.  use calcSig_wOF_ensemble.sh to compute features for all clips, stored in csv files within a specified directory structure.
      This script calls calcSig_wOF.py, which uses Temporal Segment Networks code to compute global_pool features for CNN defined in Caffe. If other options are desired, call calcSig_wOF.py directly, using calls like those in calcSig_wOF_ensemble.sh.

3.  create_db.py can create the postgres "features", "cnn_streams", and "video_clips" tables, as well as load the "cnn_streams" table.  The cnn_streams table is just a lookup table, whose sole purpose is to avoid problems with possibly spelling or capitalizing stream names in different ways.

4.   load_db.py loads the "features" table, given csv files with features from steps 1 and 2.

5.  compute_similarities.py computes similarity values for the similarity between a given reference clip and all other clips.
      All of the code in this repository is a work in progress, but this file in particular is currently being coded.

### Detailed command line instructions
1.    python tools/build_wof_clips.py  SRC_FOLDER  OUT_FOLDER    num_worker  NUM_WORKER    new_width 340 --new_height 256 2>local/errors.log
         *	SRC_FOLDER (e.g. UCF-101_test/) is the folder with the videos
         *	OUT_FOLDER (e.g. UCF-101_test_warp_frames/) is the output folder that will be populated with the frames and warped optical flow images.  There is a subdirectory for each video, holding the jpg frames and optical flow images for that video.
         *	NUM_WORKER (e.g. = 16, the default) is the number of CPU's to spread the work across. For demeter, 16 seems to be a good number if there are 16 or more videos.  Higher is not necessarily faster, as getting files on and off the GPU's can  be rate limiting.  There will not be more cpu processes than the number of videos, so NUM_WORKER greater than the number of videos has no effect.
         *	Errors.log can be substituted with a more specific name.  The program will not overwrite an existing file.
         *	Run in temporal segment networks directory within the conda TSN environment (or equivalent virtualenv).

2.	bash calcSig_wOF_ensemble.sh SRC_FOLDER_2  OUT_FOLDER_2  NUM_WORKER
        * SRC_FOLDER_2 is the folder with jpeg rgb and warped optical flow images. Normally this folder is the same as OUT_FOLDER from step 1.
        * OUT_FOLDER_2 is the root folder for the computed features
        * NUM_WORKER is the number of GPU processes
3.  python create_db.py
    * In the app, the models and migrations do this job.  I am keeping create_db.py here because it is a record of the db spec I am using.
4.  python load_db.py
    *  This program loads the "video_clips" and "features" tables.  Currently the parent directory for the features needs to be typed into load_db.py,
    but I will change that so it is an argument of load_db.py.  Normally parent_dir is the same as OUT_FOLDER_2 from step 2.

5. compute_similarities.py contains the following:
    * compute_similarities(ref_video, ref_clip, search_set='all', streams=('rgb', 'warped_optical_flow'), feature_name='global_pool', clip_duration=10)
    _Output:_
    *       avgd_similarities = {video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]}}
    * compute_score(similarities, weights={'rgb':1.0, 'warped_optical_flow':1.5})
    _Output:_
    *       scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
    * determine_matches(scores, threshold=0.8)
    _Output:_
    *       match_indicator: {<video_clip_id>: <True or False>}
    * optimize_weights(similarities, user_matches, streams=('rgb', 'warped_optical_flow'))
    _Outputs:_
    *       scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
            new_weights: {<stream>: weight}  there should be an entry for every item in streams.
            threshold_optimum: real value of computed threshold to use to separate matches from non-matches

    python compute_similarities.py:
     * will run the code for a default video name and reference clip hard coded in "main".  I am using this for testing, it is not meant
    to be something the app does.

## Environment Variables

- API_CLIENT_USERNAME
- API_CLIENT_PASSWORD
