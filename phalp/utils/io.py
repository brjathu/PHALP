import glob
import itertools
import os

import cv2
import joblib
import torchvision
from pytube import YouTube

from phalp.utils import get_pylogger
from phalp.utils.utils import FrameExtractor

log = get_pylogger(__name__)

class IO_Manager():
    '''
    Class used for loading and saving videos.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_fps = cfg.render.fps
        self.video = None

    def get_frames_from_source(self):

        # source path can be a video, or a directory of images, youtube link, or a pkl file with keys as frames
        source_path = self.cfg.video.source

        # {key: frame name, value: {"gt_bbox": None, "extra data": None}}
        additional_data = {}

        # check for youtube video
        if(source_path.startswith("https://") or source_path.startswith("http://")):

            video_name = source_path[-11:]
            os.makedirs(self.cfg.video.output_dir + "/_DEMO/" + video_name, exist_ok=True)

            # if the path is a youtube link, then download the video
            youtube_video = YouTube(source_path)
            # log.info(f'Youtube Title: {youtube_video.title}')
            # log.info(f'Video Duration: {youtube_video.length / 60:.2f} minutes')
            youtube_video.streams.get_highest_resolution().download(output_path = self.cfg.video.output_dir + "/_DEMO/" + video_name, filename="youtube.mp4")
            source_path = self.cfg.video.output_dir + "/_DEMO/" + video_name + "/youtube.mp4"


        if(source_path.endswith(".mp4")):
            # find a proper video name based on the source path
            video_name = source_path.split('/')[-1].split('.')[0]
            os.system("rm -rf " + self.cfg.video.output_dir + "/_DEMO/" + video_name + "/img/")
            if(self.cfg.video.extract_video):
                os.makedirs(self.cfg.video.output_dir + "/_DEMO/" + video_name, exist_ok=True)
                os.makedirs(self.cfg.video.output_dir + "/_DEMO/" + video_name + "/img", exist_ok=True)

                fe = FrameExtractor(source_path)
                log.info('Number of frames: ' + str(fe.n_frames))
                fe.extract_frames(every_x_frame=1, img_name='', dest_path= self.cfg.video.output_dir + "/_DEMO/" + video_name + "/img/", img_ext="." + self.cfg.EXTRA.FRAME_FORMAT, start_frame=self.cfg.video.start_frame, end_frame=self.cfg.video.end_frame)
                list_of_frames = sorted(glob.glob(self.cfg.video.output_dir + "/_DEMO/" + video_name + "/img/*." + self.cfg.EXTRA.FRAME_FORMAT))
            else:
                start_time, end_time = int(self.cfg.video.start_time[:-1]), int(self.cfg.video.end_time[:-1])
                try:
                    # TODO: check if torchvision is compiled from source
                    raise Exception("torchvision error")
                    # https://github.com/pytorch/vision/issues/3188
                    reader = torchvision.io.VideoReader(source_path, "video")
                    list_of_frames = []
                    for frame in itertools.takewhile(lambda x: x['pts'] <= end_time, reader.seek(start_time)):
                        list_of_frames.append(frame['data'])
                except:
                    log.warning("torchvision is NOT compliled from source!!!")

                    stamps_PTS = torchvision.io.read_video_timestamps(source_path, pts_unit='pts')
                    stamps_SEC = torchvision.io.read_video_timestamps(source_path, pts_unit='sec')

                    index_start = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - start_time))
                    index_end = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - end_time))

                    if(index_start==index_end and index_start==0):
                        index_end += 1
                    elif(index_start==index_end and index_start==len(stamps_SEC[0])-1):
                        index_start -= 1

                    # Extract the corresponding presentation timestamps from stamps_PTS
                    list_of_frames = [(source_path, i) for i in stamps_PTS[0][index_start:index_end]]

        # read from image folder
        elif(os.path.isdir(source_path)):
            video_name = source_path.split('/')[-1]
            list_of_frames = sorted(glob.glob(source_path + "/*." + self.cfg.EXTRA.FRAME_FORMAT))

        # pkl files are used to track ground truth videos with given bounding box
        # these gt_id, gt_bbox will be stored in additional_data, ground truth bbox should be in the format of [x1, y1, w, h]
        elif os.path.isfile(source_path) and source_path.endswith(".pkl"):
            gt_data = joblib.load(source_path)
            video_name = source_path.split('/')[-1].split('.')[0]
            list_of_frames = [os.path.join(self.cfg.video.base_path, i) for i in sorted(list(gt_data.keys()))]

            # for adding gt bbox for detection
            # the main key is the bbox, rest (class label, track id) are in extra data.
            for frame_name in list_of_frames:
                frame_id = frame_name.split('/')[-1]
                if len(gt_data[frame_id]['gt_bbox'])>0:
                    additional_data[frame_name] = gt_data[frame_id]
                    '''
                    gt_data structure:
                    gt_data[frame_id] = {
                                            "gt_bbox": gt_boxes,
                                            "extra_data": {
                                                "gt_class": [],
                                                "gt_track_id": [],
                                            }
                                        }
                    '''
        else:
            raise Exception("Invalid source path")

        io_data = {
            "list_of_frames": list_of_frames,
            "additional_data": additional_data,
            "video_name": video_name,
        }

        return io_data

    @staticmethod
    def read_frame(frame_path):
        frame = None
        # frame path can be either a path to an image or a list of [video_path, frame_id in pts]
        if(isinstance(frame_path, tuple)):
            frame = torchvision.io.read_video(frame_path[0], pts_unit='pts', start_pts=frame_path[1], end_pts=frame_path[1]+1)[0][0]
            frame = frame.numpy()[:, :, ::-1]
        elif(isinstance(frame_path, str)):
            frame = cv2.imread(frame_path)
        else:
            raise Exception("Invalid frame path")

        return frame

    @staticmethod
    def read_from_video_pts(video_path, frame_pts):
        frame = torchvision.io.read_video(video_path, pts_unit='pts', start_pts=frame_pts, end_pts=frame_pts+1)[0][0]
        frame = frame.numpy()[:, :, ::-1]
        return frame

    def reset(self):
        self.video = None

    def save_video(self, video_path, rendered_, f_size, t=0):
        if(t==0):
            self.video = {
                "video": cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.output_fps, frameSize=f_size),
                "path" : video_path,
                }
        if(self.video is None):
            raise Exception("Video is not initialized")
        self.video["video"].write(rendered_)

    def close_video(self):
        if(self.video is not None):
            self.video["video"].release()
            if(self.cfg.video.useffmpeg):
                ret = os.system("ffmpeg -hide_banner -loglevel error -y -i {} {}".format(self.video["path"], self.video["path"].replace(".mp4", "_compressed.mp4")))
                # Delete if successful
                if(ret == 0):
                    os.system("rm {}".format(self.video["path"]))
            self.video = None
