import pyrealsense2 as rs
import numpy as np
import cv2
from utils.datasets import LoadStreams, LoadImages, LoadRealSense2

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
Lo = LoadRealSense2()
print(type(Lo))
# pipeline.start(config)


while True:
    frames = Lo.pipe.wait_for_frames()
    aligned1 = Lo.aligned(frames)
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    aligned_depth_frame = frames.get_depth_frame()
    print("depth value in m:{0}".format(aligned_depth_frame.get_distance(320, 240)))

    # print(aligned_depth_frame)

    

