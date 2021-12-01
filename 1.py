import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# ###
import numpy as np
# 建立一張 512x512 的 RGB 圖片（黑色）
img1 = np.zeros((1028, 1028, 3), np.uint8)
# text
text = '== Nutrition Facts =='
text0 = 'Rice(100G)'
text1 = 'Fat          Carbs          Protein'
text2 = '0.3 g         41 g            3.1g'
text3 = 'Total:'
text4 = '0.3 g         41 g            3.1g'
text5 =  'Calories'
text7 = '183 kcal'

cv2.putText(img1, text, (0, 90), cv2.FONT_HERSHEY_SIMPLEX,2.8, (0, 255, 255),2, cv2.LINE_AA)
cv2.putText(img1, text0, (330, 180), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img1, text1, (0, 250), cv2.FONT_HERSHEY_SIMPLEX,1.8, (0, 255, 255),2, cv2.LINE_AA)
cv2.putText(img1, text2, (0, 330), cv2.FONT_HERSHEY_SIMPLEX,1.7, (0, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img1, text3, (0, 450), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255),2, cv2.LINE_AA)
cv2.putText(img1, text4, (0, 530), cv2.FONT_HERSHEY_SIMPLEX,1.7, (0, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img1, text5, (50, 610), cv2.FONT_HERSHEY_SIMPLEX,1.7, (0, 255, 255),2, cv2.LINE_AA)
cv2.putText(img1, text7, (750, 610), cv2.FONT_HERSHEY_SIMPLEX,1.7, (0, 255, 255),2, cv2.LINE_AA)
# resize image
img1 = cv2.resize(img1, (500, 500))

# cv2.imshow("test1",img)

from testdetect import detect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            print('2')
            detect()

