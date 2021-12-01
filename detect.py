import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadRealSense2
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pyrealsense2 as rs
import schedule  


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


data = []
object_name_list = []
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') or source ==('realsense')

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if source=='realsense':
            dataset = LoadRealSense2()
        else:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #print(f'{names[int(c)]}test12')

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                       
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        # print('bounding box is ', x1, y1, x2, y2)

                        

                        ########################librealsense 中心
                        # frames = dataset.pipe.wait_for_frames()
                        # aligned1 = dataset.aligned(frames)
                        # align = rs.align(rs.stream.color)
                        # frames = align.process(frames)
                        # aligned_depth_frame = frames.get_depth_frame()
                        # print("depth value in m:{0}".format(aligned_depth_frame.get_distance(320, 240)))
                        ########################加入物體中心深度
                        bbox_points=[x1, y1, x2, y2]


                        # for ii in bbox_points:
                        #     # cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)#cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
                        #     text_depth ='depth is '+ str(np.round(aligned_depth_frame.get_distance(int((x1+x2)*(1/2)), int((y1+y2)*(1/2))),3))+"m"
                        #     im0=cv2.putText(im0,text_depth,(x1,y1+40),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),2,cv2.LINE_AA)
                            ###cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)

                        #### 切割標記出來的圖片
                        ### bbox_points=[x1, y1, x2, y2]
                        for i in bbox_points:
                    	    cropped_img = im0[y1:y2, x1:x2]
                    	    cv2.imwrite("crop1.jpg", cropped_img) ###-----put the output folder path here---####
                    	    i+=1
                        
                        print('(',x1,',' ,y1,')','(' ,x2,',', y2,')')
                        data.append(bbox_points)
                        # print(x2-x1,y2-y1)
                        # print((x2-x1)*(y2-y1))                        
                        # print(cropped_img.size) ##size 要乘3 因為有rgb
                        # print((cropped_img))

                    #cv2.imwrite('test.png',cropped_img)               
                        class_index = cls
                        object_name = names[int(cls)]
                        object_name1 = object_name.split()
                        object_name_list.append(object_name1)
                        # label = label+分數
                        for x in object_name1:
                          if x == 'rice':
                            print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','183 kcal  41 g        3.1 g   0.3g\n')
                        for x in object_name1:
                          if x == 'egg':
                            print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','144 kcal  1.7 g        14 g   9.2g\n')
                        for x in object_name1:
                          if x == 'salmon':
                            print('Detected object name is',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','171 kcal  0 g        20.7 g   9.5g\n')
                        for x in object_name1:
                          if x == 'tofu':
                            print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','131 kcal  3.9 g        12.9 g   1.3g\n')
                        for x in object_name1:
                          if x == 'sauteed vegetables':
                            print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','24 kcal  4.9 g        1.4 g   0.2\n')


                        # conf 評分
                        confidence_score = conf                       
                        # 儲存圖片
                        original_img = im0
                        # resize detect image 
                        img0 = cv2.resize(im0, (500, 500))

                        # image merge
                        htitch= np.hstack((img0, img1,))
                        # image show
                        cv2.imwrite("test1.jpg",htitch)

                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()                       
                        # # # print(label)
                        # print(type(object_name))
                        print(object_name)
                        # print(object_name1)
                        #print('class index is ', class_index)
                        # print('detected object name is ', object_name)
                        # original_img = im0
                        # cropped_img = im0[y1:y2, x1:x2]
                        # cv2.imwrite('test.png',cropped_img)
            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            #print(f'{names[int(c)]}test14')
            #print(type(s))
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


# schedule.every(1).minutes.do(detect) 

# while True:  
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
    # print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # schedule.run_pending()  

            detect()
            # time.sleep(1)  

