""" calcSig_wOF.py computes video signatures comprising embedded features of the RGB and warped optical flow CNNs.
Here, the CNN's are networks (i.e. RGB or warped optical flow) trained using the Temporal Segment Networks
(TSN) work by Wang et al. (https://github.com/yjxiong/temporal-segment-networks ).
The default name for the feature blob (--featureBlob) is "global_pool", which is at the "fc-action"
layer in the TSN net. Also by default, the computed RGB signature is the average over 25 frames in
the video, and the optical flow signature is the average over 25 stacks of 5 x-direction flow images
and 5 y-direction flow images.  These follow the method used in the TSN publication to score videos.
The option --num_frame_per_video can be used to change this parameter.
"""

import sys
import os
# specify the directory for the temporal segment networks code by
# setting the environment variable TSN_ROOT, e.g. = '/data/torres/temporal-segment-networks/'
sys.path.insert(0, "$TSN_ROOT")
sys.path.insert(1, os.path.join("$TSN_ROOT", 'lib/caffe-action/python'))

import argparse
import glob
import cv2
import numpy as np
import multiprocessing
import errno
# CaffeNet objects contain learned caffe nets and functions for scoring frames
from pyActionRecog.action_caffe import CaffeNet
from pyActionRecog import parse_directory  # for parsing directories holding frames
#  output is (dir_dict, rgb_counts, flow_counts) where the counts are frame counts


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return


def build_net(net_proto, net_weights, max_gpu_id=10):
    # Note:  CaffeNet creates a caffe.TEST net, so dropout layer does not
    # drop anything and the TEST input data layer, if present, is used.
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        assert (my_id <= max_gpu_id)
        net = CaffeNet(net_proto, net_weights, my_id-1)
    else:
        gpu_id = (my_id-1) % len(gpu_list)
        net = CaffeNet(net_proto, net_weights, gpu_list[gpu_id])
    return


def eval_clip(vid):
    global net
    # f_info containing 3 dictionaries is needed: [ {vid:path}, {vid:rgb_counts}, {vid:flow_counts} ]
    # s (from streamCNN) contains the relevant dictionary for the stream being analyzed - either rgb or flow
    video_frame_path = f_info[0][vid]
    frame_cnt = f_info[ s['cnt_indexer'] ][vid]

    # analyze num_frame_per_video frames in each video, regardless of video length, per TSN paper
    step = (frame_cnt - s['stack_depth']) / (args.num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (args.num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * args.num_frame_per_video
    assert(len(frame_ticks) == args.num_frame_per_video)  # Make sure the proper number of ticks were created

    if (s['modality']=='rgb'):
        frame_features = rgbFeatureExtract(video_frame_path, frame_ticks)
    elif (s['modality']=='flow'):
        frame_features = flowFeatureExtract(video_frame_path, frame_ticks, frame_cnt, s['stack_depth'])

    # compute average over all frame features for this video.  This creates one mean array within the outer array,
    # so extract the inner array with the [0] selection.

    v_avg_feature = np.array(frame_features).mean(axis=0)[0]
    print( 'video {} for {} modality done'.format(vid,s['modality']) )
    sys.stdin.flush()
    return v_avg_feature


def rgbFeatureExtract( video_frame_path, frame_ticks):
    global net
    frame_features = []
    for tick in frame_ticks:
        name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
        frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
        net.predict_single_frame([frame, ], score_name, frame_size=(340, 256))
        frame_features.append( net._net.blobs[args.featureBlob].data[0].reshape(1, -1).tolist() )
    return frame_features


def flowFeatureExtract(video_frame_path, frame_ticks, frame_cnt, stk_depth):
    global net
    # s (from streamCNN) contains the relevant dictionary for the stream being analyzed - either rgb or flow
    frame_features = []
    for tick in frame_ticks:
        frame_idx = [ min(frame_cnt, tick+offset) for offset in range( stk_depth ) ]
        flow_stack = []
        for idx in frame_idx:
            x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
            y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
            flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
            flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
        net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
        frame_features.append( net._net.blobs[args.featureBlob].data[0].reshape(1, -1).tolist() )
    return frame_features


def writeFeatures(clip_list, video_path):
    video = video_path.split('/')[-2]  # [-2] because [-1] is an empty string since path ends with a '/'
    f_output_dir = os.path.join(args.outFeatures_dir, video, args.modelname)
    make_sure_path_exists(f_output_dir)
    for mode in [('rgb', rgbFeature), ('warped_optical_flow', flowFeature)]:
        if mode[0] == 'rgb':
            net_weights_file = args.net_weights_rgb
        else:
            net_weights_file = args.net_weights_flow
        header_txt = 'video =' + video + ', video url =' + video_path + ', CNN stream =' + mode[0] \
                     + ', feature blob =' + args.featureBlob + ', caffe model =' + net_weights_file
        outfile = os.path.join(f_output_dir, mode[0] + "_" + args.featureBlob + "_features.csv")
        with open(outfile, mode='w') as fout:
            fout.write(header_txt + "\n")
            for i, vid in enumerate(clip_list):
                clip_no = int(vid[-4:])
                row = str(clip_no) + "," + ",".join(map(str, mode[1][i]))
                fout.write(row + "\n")
    return


#************************************************************************************
##   Main program
##

#  Parse the arguments of evalSig
#
# example:  python tools/evalSig.py testVideo_frames/
#           models/ucf101/tsn_bn_inception_rgb_deploy.prototxt
#           models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel
#           models/ucf101/tsn_bn_inception_flow_deploy.prototxt
#           models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel
#           --num_worker 24 --outFeatures_dir testVideo_features/ --modelname UCF101_model
#
# Note:  num_worker is the number of processes, each of which starts with a CPU and then grabs
#        gpu resources.  num_worker can be larger than the number of available GPUs, but in
#        that case, the --gpus option must explicitly say what GPU numbers are available.
#        Otherwise, the program will increment the gpu number, rather than load up multiple jobs
#        on a GPU.
#
parser = argparse.ArgumentParser()
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto_rgb', type=str)
parser.add_argument('net_weights_rgb', type=str)
parser.add_argument('net_proto_flow', type=str)
parser.add_argument('net_weights_flow', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--num_frame_per_video', type=int, default=25, help="number of frames to evaluate in each video")
parser.add_argument('--num_worker', type=int, default=1, help="number of cpu cores to use. If > number of GPUs, \
                                     set --gpus explicitly so multiple jobs get loaded on each GPU")
parser.add_argument("--gpus", type=int, nargs='+', default=None,
                    help='specify list of gpu to use, e.g. "--gpus 0 1 2 3" or gpu will be set to process number')
parser.add_argument('--outFeatures_dir', type=str, default=None,  help='Specify directory to write out feature files')
parser.add_argument('--delimiter', type=str, default=',', help='delimiter used in feature files')
parser.add_argument('--modelname', type=str, default=None)
parser.add_argument('--featureBlob', type=str, default='global_pool',
                        help='name of blob in caffe net to extract as a feature')
parser.add_argument('--featureBlob_size', type=int, default=1024,
                        help='expected size of blob in caffe net to be extracted as a feature')

args = parser.parse_args()
if ( args.modelname == None ):
    args.modelname = args.net_weights_rgb.split('/')[-1][:-11] + '_' + args.net_weights_flow.split('/')[-1][:-11]
print (args)
gpu_list = args.gpus
frame_path = args.frame_path
score_name = 'fc-action'
streamCNN = [ { 'modality':'rgb', 'net_proto': args.net_proto_rgb, 'net_weights': args.net_weights_rgb,
               'cnt_indexer':1, 'stack_depth':1 },
              { 'modality':'flow', 'net_proto': args.net_proto_flow, 'net_weights': args.net_weights_flow,
               'cnt_indexer':2, 'stack_depth':5 }
            ]
# Loop over each video in args.frame_path
if frame_path[-1] != '/':
    frame_path = frame_path + '/'
videos = glob.glob(frame_path + '*/')

for video_path in videos:
    # build necessary information about video names, directories of frames, and number of rgb and flow files
    #f_info contains 3 dictionaries: [ {vid:path}, {vid:rgb_counts}, {vid:flow_counts} ]
    f_info = parse_directory(video_path, args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)
    eval_clip_list = list(f_info[0])
    eval_clip_list = sorted(eval_clip_list, key=lambda clip: int(clip[-4:]))  # put clips in order
    num_videos = len(eval_clip_list)

    # extract the desired caffe net features and store in files
    if args.num_worker > 1:
        for s in streamCNN:
            pool = multiprocessing.Pool( args.num_worker, initializer=build_net,
            initargs=( s['net_proto'], s['net_weights'] ) )
            s['clip_features'] = pool.map(eval_clip, eval_clip_list)
            pool.close()
            pool.join()
    else:
        for s in streamCNN:
            build_net( s['net_proto'], s['net_weights'] )
            s['clip_features'] = map(eval_clip, eval_clip_list)

    # Store the average caffe net rgb and flow features for the videos in a file
    rgbFeature = streamCNN[0]['clip_features']
    flowFeature = streamCNN[1]['clip_features']
    numFeatures = len(rgbFeature[0])
    assert numFeatures==args.featureBlob_size
    writeFeatures(eval_clip_list, video_path)
