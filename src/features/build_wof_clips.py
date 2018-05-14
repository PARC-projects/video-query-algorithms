#__author__ = 'yjxiong'
# 1/11/2018: Revised by Frank Torres to compute rgb and warped optical flow frames
# for use in video query, and separate the jpg files into folders for each clip.
# Default video type is changed from .avi to .mp4.
# The end result is the directory out_dir with subdirectories for each video and
# subdirectories below that for each clip, with the frames and optical flow images in
# the lowest level subdirectories.

import sys
# specify the directory for the temporal segment networks code by
# setting the environment variable TSN_ROOT, e.g. = '/data/torres/temporal-segment-networks/'
sys.path.insert(0, "$TSN_ROOT")
import os
import glob
import shutil
import cv2
from multiprocessing import Pool, current_process
import argparse
out_path = ''


def dump_frames(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    ret, frame = video.read()  # skip the initial blank frame
    assert ret
    # fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) - 1
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    icount = 1
    while (True):
        ret, frame = video.read()
        if not ret:
            break
        if new_size != (0,0):
            frame = cv2.resize(frame, new_size)
        cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, icount), frame)
        access_path = '{}/{:05d}.jpg'.format(vid_name, icount)
        file_list.append(access_path)
        icount += 1
    print 'rgb for {} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return file_list

def run_warp_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU + START_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        vid_path, flow_x_path, flow_y_path, dev_id, out_format)

    os.system(cmd)
    print 'warp on {} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True

def create_clip(vid_path, out_path, frames_per_clip=150, frames_per_second=15):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    rgbFrames = glob.glob(os.path.join(out_path, vid_name, 'img*.jpg'))
    nFrames = len(rgbFrames)
    nclips = int(nFrames / frames_per_clip)  # truncates collection of frames if not an integer multiple of frames_per_clip
    out_video_path = os.path.join(out_path, vid_name)
    for n in range(nclips):
        frames = list(range(n * frames_per_clip + 1, (n + 1) * frames_per_clip + 1))
        nClipDir = os.path.join(out_video_path, 'clip_{:04d}'.format(n + 1))
        os.mkdir(nClipDir)

        for iframe in frames:
            rgbFile = os.path.join(out_video_path, 'img_{:05d}.jpg'.format(iframe))
            flowxFile = os.path.join(out_video_path, 'flow_x_{:05d}.jpg'.format(iframe))
            flowyFile = os.path.join(out_video_path, 'flow_y_{:05d}.jpg'.format(iframe))

            newFrame = iframe - n * frames_per_clip
            rgbDestFile = os.path.join(nClipDir, 'img_{:05d}.jpg'.format(newFrame))
            flowxDestFile = os.path.join(nClipDir, 'flow_x_{:05d}.jpg'.format(newFrame))
            flowyDestFile = os.path.join(nClipDir, 'flow_y_{:05d}.jpg'.format(newFrame))
            shutil.move(rgbFile, rgbDestFile)
            shutil.move(flowxFile, flowxDestFile)
            shutil.move(flowyFile, flowyDestFile)

    # Make a short clip out of any remaining frames, as long as the clip is at least 2 sec
    rgbFrames_remaining = glob.glob(os.path.join(out_path, vid_name, 'img*.jpg'))
    nFrames_remaining = len(rgbFrames_remaining)
    if nFrames_remaining >= 2*frames_per_second:
        nClipDir = os.path.join(out_video_path, 'clip_{:04d}'.format(nclips+1))
        os.mkdir(nClipDir)

    frames = list(range(nclips * frames_per_clip + 1,
                        nclips * frames_per_clip + nFrames_remaining + 1))
    for iframe in frames:
        rgbFile = os.path.join(out_video_path, 'img_{:05d}.jpg'.format(iframe))
        flowxFile = os.path.join(out_video_path, 'flow_x_{:05d}.jpg'.format(iframe))
        flowyFile = os.path.join(out_video_path, 'flow_y_{:05d}.jpg'.format(iframe))
        if nFrames_remaining >= 2*frames_per_second:
            newFrame = iframe - nclips * frames_per_clip
            rgbDestFile = os.path.join(nClipDir, 'img_{:05d}.jpg'.format(newFrame))
            flowxDestFile = os.path.join(nClipDir, 'flow_x_{:05d}.jpg'.format(newFrame))
            flowyDestFile = os.path.join(nClipDir, 'flow_y_{:05d}.jpg'.format(newFrame))
            shutil.move(rgbFile, rgbDestFile)
            shutil.move(flowxFile, flowxDestFile)
            shutil.move(flowyFile, flowyDestFile)
        else:
            os.remove(rgbFile)
            os.remove(flowxFile)
            os.remove(flowyFile)

    return

if __name__ == '__main__':
    flow_type = "warp_tvl1"

    parser = argparse.ArgumentParser(description="Extract rgb and warped optical flow frames")
    parser.add_argument("src_dir",
                        help="directory with video files")
    parser.add_argument("out_dir")
    parser.add_argument("--fps", type=int, default=15, help="frames per second, default = 15")
    parser.add_argument("--clip_time", type=int, default=10, help="clip time in seconds, default = 10")
    parser.add_argument("--num_worker", type=int, default=16, help="CPU workers, default=16")
    #parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--df_path", type=str, default='./lib/dense_flow/', help='path to the dense_flow toolbox, '
                                                                                 'default = ./lib/dense_flow/')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='format of output, default=dir')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'], help='video file extensions, '
                                                                                      'default = mp4')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU, default = 8')
    parser.add_argument("--starting_gpu", type=int, default=0, help='ID of first GPU to use, default = 0')

# initialization
    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    frames_per_clip = args.clip_time * args.fps
    num_worker = args.num_worker
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    assert new_size == (0,0) or (new_size[0] != 0 and new_size[1] != 0)
    NUM_GPU = args.num_gpu
    START_GPU = args.starting_gpu

    if not os.path.isdir(out_path):
        print "creating folder: "+out_path
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*.'+ext)
    print "number of videos found = {}".format(len(vid_list) )

# create warped optical flow images and write to file
    pool = Pool(num_worker)
    pool.map(run_warp_optical_flow, zip(vid_list, xrange(len(vid_list))))
    pool.close()
    pool.join()

# extract rgb frames from video and write to file
    pool_rgb = Pool(num_worker)
    pool_rgb.map(dump_frames, zip(vid_list, xrange(len(vid_list))))
    pool_rgb.close()
    pool_rgb.join()

# reorganize frames and warped optical flow images into clip directories
    for vid_path in vid_list:
        create_clip(vid_path, out_path, frames_per_clip=frames_per_clip,
                   frames_per_second=args.fps)
