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

# ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import schedule
t1 = time.time()

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


coordinate = []
object_name_list = []
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') or source ==('realsense')
    global x1, y1, x2, y2

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
                        global bbox_points
                        
                        bbox_points=[x1, y1, x2, y2]
                        coordinate.append(bbox_points)
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
                        
                        # print('(',x1,',' ,y1,')','(' ,x2,',', y2,')')
                        # print(x2-x1,y2-y1)
                        # print((x2-x1)*(y2-y1))                        
                        # print(cropped_img.size) ##size 要乘3 因為有rgb
                        # print((cropped_img))

                    #cv2.imwrite('test.png',cropped_img)    
                        # global object_name1
                        # global object_name       
                        class_index = cls
                        object_name = names[int(cls)]
                        object_name1 = object_name.split()
                        object_name_list.append(object_name)
                        # object_name_list1.append(object_name1)
                        # print(object_name)
                        # return object_name
                        # return object_name1
                        # return bbox_points

                        # label = label+分數
                        # for x in object_name1:
                        #   if x == 'rice':
                        #     print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','183 kcal  41 g        3.1 g   0.3g\n')
                        # for x in object_name1:
                        #   if x == 'egg':
                        #     print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','144 kcal  1.7 g        14 g   9.2g\n')
                        # for x in object_name1:
                        #   if x == 'salmon':
                        #     print('Detected object name is',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','171 kcal  0 g        20.7 g   9.5g\n')
                        # for x in object_name1:
                        #   if x == 'tofu':
                        #     print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','131 kcal  3.9 g        12.9 g   1.3g\n')
                        # for x in object_name1:
                        #   if x == 'sauteed vegetables':
                        #     print('Detected object name is:',label,'\n',object_name,'(100 g)','\n','========= Nutrition Facts =========','\n','總卡路里  碳水化合物  蛋白質  脂肪','\n','24 kcal  4.9 g        1.4 g   0.2\n')


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
                        # print(object_name1)
                        #print('class index is ', class_index)
                        # print('detected object name is ', object_name1)
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
    def volume(food,vol):
    ## 密度
        rice_density = 0.73 
        egg_density = 0.6
        brocoli_density = 0.45
        Shrimp_density = 0.77
        sanpper_density = 1.07
        tofu_density = 1.04
        if food == 'broccoli':
            brocoli_weight = vol*brocoli_density
            brocoli_weight = round(brocoli_weight, 2)
            ### 除100
            brocoli_weight1 = brocoli_weight/100
            Calories = brocoli_weight1*23
            Calories = round(Calories, 1)
            fat = brocoli_weight1*0.1 ##脂肪
            fat = round(fat, 1)
            carbohydrate = brocoli_weight1*4.5
            carbohydrate = round(carbohydrate, 1)
            protein = brocoli_weight1*1.8
            protein = round(protein, 1)
            print(food,'(',brocoli_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
    
    
        if food == 'rice':
            rice_weight = vol*rice_density
            rice_weight = round(rice_weight, 2)########## round 為四捨五入
            ### 除100
            rice_weight1 = rice_weight/100
            Calories = rice_weight1*183
            Calories = round(Calories, 1)
            fat = rice_weight1*0.3 ##脂肪
            fat = round(fat, 1)
            carbohydrate = rice_weight1*41
            carbohydrate = round(carbohydrate, 1)
            protein = rice_weight1*3.1
            protein = round(protein, 1)
        
            print(food,'(',rice_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
    
    
        if food == 'Shrimp':
            Shrimp_weight = vol*Shrimp_density
            Shrimp_weight = round(Shrimp_weight, 2)
            ### 除100
            Shrimp_weight1 = Shrimp_weight/100
            Calories = Shrimp_weight1*122
            Calories = round(Calories, 1)
            fat = Shrimp_weight1*4.2 ##脂肪
            fat = round(fat, 1)
            carbohydrate = Shrimp_weight1*2.7
            carbohydrate = round(carbohydrate, 1)
            protein = Shrimp_weight1*19.9
            protein = round(protein, 1)
        
            print(food,'(',Shrimp_weight-3,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
    
        if food == 'egg':
            egg_weight = vol*egg_density
            egg_weight = round(egg_weight, 2)
            ### 除100
            egg_weight1 = egg_weight/100
            Calories = egg_weight1*144
            Calories = round(Calories, 1)
            fat = egg_weight1*9.2 ##脂肪
            fat = round(fat, 1)
            carbohydrate = egg_weight1*1.7
            carbohydrate = round(carbohydrate, 1)
            protein = egg_weight1*14
            protein = round(protein, 1)
        
            print(food,'(',egg_weight+16,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')

        if food == 'snapper':
            snapper_weight = vol*snapper_density
            snapper_weight = round(snapper_weight, 2)
            ### 除100
            snapper_weight1 = snapper_weight/100
            Calories = snapper_weight1*110
            Calories = round(Calories, 1)
            fat = snapper_weight1*3.6 ##脂肪
            fat = round(fat, 1)
            carbohydrate = snapper_weight1*2.5
            carbohydrate = round(carbohydrate, 1)
            protein = snapper_weight1*18.2
            protein = round(protein, 1)
            print(food,'(',snapper_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
    
        if food == 'tofu':
            tofu_weight = vol*tofu_density
            tofu_weight = round(tofu_weight, 2)
            ### 除100
            tofu_weight1 = tofu_weight/100
            Calories = tofu_weight1*23
            Calories = round(Calories, 1)
            fat = tofu_weight1*0.1 ##脂肪
            fat = round(fat, 1)
            carbohydrate = tofu_weight1*4.5
            carbohydrate = round(carbohydrate, 1)
            protein = tofu_weight1*1.8
            protein = round(protein, 1)
            print(food,'(',tofu_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
    
#######################################################################################################
## 補零
    img_gray0 = cv2.imread('/home/kevin/yolov5work/yolov5/data/images/0.png', cv2.IMREAD_GRAYSCALE)

    # c , d  = img_gray0.shape
    data2=[]##補零用
    data1=[]##存分割後圖片座標

    # for i in range(c):
    #     for j in range(d):
    #         data2.append((i,j))
    original_depth0 = np.load("/home/kevin/yolov5work/yolov5/data/images/0.npy")##原始
    original_depth1 = np.load("/home/kevin/yolov5work/yolov5/data/images/1.npy")##後來

    # for i in range(len(data2)):
            
        
    #     original_depth01 = original_depth0[data2[i]]
    #     original_depth01 = float(original_depth01)
    #     original_depth11 = original_depth1[data2[i]]
    #     original_depth11 = float(original_depth11)         
    #     # print(i,truth_depth1)
        
    #     if original_depth01 == 0.0:
    #         original_depth0[data2[i]] = original_depth0[data2[i-1]]
    #     if original_depth11 == 0.0:
    #         original_depth1[data2[i]] = original_depth1[data2[i-1]]
    # ## 補零
    #######################################################################################################
    # # 處理bowl
    # 裝bowl的list
    bowll = []
    bowl_coordinate = []
    # print(object_name_list,coordinate,'111111111111111111')
    for q in range(len(object_name_list)):
        if object_name_list[q] == 'bowl':
            bowll.append(object_name_list[q])
            bowl_coordinate.append(coordinate[q])
            coordinate[q] = (0,0,0,0)
    # print(object_name_list)
    bowl_coordinate_array = np.array(bowl_coordinate) 
    
    ##刪bowl
    object_name_list_no_bowl = list(filter(('bowl').__ne__, object_name_list))##https://www.delftstack.com/zh-tw/howto/python/python-list-remove-all/
    ##刪bowl座標
    coordinate_no_bowl = list(filter(((0,0,0,0)).__ne__, coordinate))       
    print(object_name_list_no_bowl)
    print(coordinate_no_bowl)   
    print(bowll)
    print(bowl_coordinate)   

    #######################################################################################################
     

    #######################################################################################################
    total = []## 深度差
    local = []## 有深度差的座標
    local_depth = []## 座標原始深度

    # print(len(object_name_list))
    # print(coordinate)
    coordinate_no_bowl_array = np.array(coordinate_no_bowl)
    print(bowl_coordinate_array)  

    # print(coordinate_array,object_name_list,'333333333333333333333333')

    # for i in range(len(coordinate_array)):
    #     for j in range(4):
    #         print(coordinate_array[i][j])
    for g in range(len(bowll)):
        ##  找出座標點
        x1, y1, x2, y2 = bowl_coordinate_array[g][0],bowl_coordinate_array[g][1],bowl_coordinate_array[g][2],bowl_coordinate_array[g][3]
        cropped_img = img_gray0[y1:y2, x1:x2]
        a , b = cropped_img.shape
        # print(cropped_img.shape)
        for i in range(a):
            for j in range(b):
                data1.append((i,j))
            
        for i in range(len(data1)):
            # print(len(data1),i)
            
            later_depth0 = original_depth0[ y1 : y2, x1: x2 ]##切割
            later_depth1 = original_depth1[ y1 : y2, x1: x2 ]##切割

            truth_depth1 = f'{later_depth1[data1[i]]}'
            truth_depth0 = f'{later_depth0[data1[i]]}'
            truth_depth1 = float(truth_depth1)
            truth_depth0 = float(truth_depth0)
            
            if (truth_depth1-truth_depth0) > 10 and truth_depth1!=0 and  truth_depth0!= 0 :##深度大於某數在計算 確保誤差
            # if (truth_depth1-truth_depth0) > 10 要搭配修補
                
                total.append(truth_depth1-truth_depth0)##深度差集合
                f'{local.append(data1[i])}'##有深度差的座標集合
        for i in range(len(total)):###有深度差大於某數的次數 也為像素數
            
            # print('第二次',len(total),i,local[i])
            truth_depth0 = f'{original_depth0[local[i]]}'###有深度差的座標原始深度  
            local_depth.append(truth_depth0)#座標原始深度集合
        
        local_depth[:] = [float(x) for x in local_depth]##https://whhnote.blogspot.com/2010/12/python-list.html
        np_local_depth = np.array(local_depth)
        ##https://ithelp.ithome.com.tw/articles/10226199
        np_local_depth = np_local_depth/10
        length = 0.00166187 * np_local_depth + 0.001747800933376259
        area = length * length
        # print(area)
        total[:] = [float(x) for x in total]
        np_total = np.array(total)
        np_total = np_total/10
        # print(area.shape)
        # print(np_total.shape)
        # print('前',len(data1))

        vol = area * np_total## 體積
        volume(object_name_list[g],sum(vol))
        print(bowll)
        print(data1)
        print(len())
        # print(object_name_list[g])
        # print(sum(vol),object_name_list[g],'xxxxxxxxxxxxxxxxx')
        # print(object_name_list[g],'vol = ',sum(vol))
        ### 將list清空
        # print(len(local))
   
        data1.clear()
        total.clear()
        local.clear()
        local_depth.clear()
        # print('後',len(data1))

        
    

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







################################################################################################################
# def volume(food,vol):
#     ## 密度
#     rice_density = 0.73 
#     egg_density = 0.6
#     brocoli_density = 0.45
#     Shrimp_density = 0.77
#     sanpper_density = 1.07
#     tofu_density = 1.04
#     if food == 'brocoli':
#         brocoli_weight = vol*brocoli_density
#         brocoli_weight = round(brocoli_weight, 2)
#         ### 除100
#         brocoli_weight1 = brocoli_weight/100
#         Calories = brocoli_weight1*23
#         Calories = round(Calories, 1)
#         fat = brocoli_weight1*0.1 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = brocoli_weight1*4.5
#         carbohydrate = round(carbohydrate, 1)
#         protein = brocoli_weight1*1.8
#         protein = round(protein, 1)
#         print(food,'(',brocoli_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
  
  
#     if food == 'rice':
#         rice_weight = vol*rice_density
#         rice_weight = round(rice_weight, 2)########## round 為四捨五入
#         ### 除100
#         rice_weight1 = rice_weight/100
#         Calories = rice_weight1*183
#         Calories = round(Calories, 1)
#         fat = rice_weight1*0.3 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = rice_weight1*41
#         carbohydrate = round(carbohydrate, 1)
#         protein = rice_weight1*3.1
#         protein = round(protein, 1)
      
#         print(food,'(',rice_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
  
  
#     if food == 'Shrimp':
#         Shrimp_weight = vol*Shrimp_density
#         Shrimp_weight = round(Shrimp_weight, 2)
#         ### 除100
#         Shrimp_weight1 = Shrimp_weight/100
#         Calories = Shrimp_weight1*122
#         Calories = round(Calories, 1)
#         fat = Shrimp_weight1*4.2 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = Shrimp_weight1*2.7
#         carbohydrate = round(carbohydrate, 1)
#         protein = Shrimp_weight1*19.9
#         protein = round(protein, 1)
      
#         print(food,'(',Shrimp_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
   
#     if food == 'egg':
#         egg_weight = vol*egg_density
#         egg_weight = round(egg_weight, 2)
#         ### 除100
#         egg_weight1 = egg_weight/100
#         Calories = egg_weight1*144
#         Calories = round(Calories, 1)
#         fat = egg_weight1*9.2 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = egg_weight1*1.7
#         carbohydrate = round(carbohydrate, 1)
#         protein = egg_weight1*14
#         protein = round(protein, 1)
      
#         print(food,'(',egg_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')

#     if food == 'snapper':
#         snapper_weight = vol*snapper_density
#         snapper_weight = round(snapper_weight, 2)
#         ### 除100
#         snapper_weight1 = snapper_weight/100
#         Calories = snapper_weight1*110
#         Calories = round(Calories, 1)
#         fat = snapper_weight1*3.6 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = snapper_weight1*2.5
#         carbohydrate = round(carbohydrate, 1)
#         protein = snapper_weight1*18.2
#         protein = round(protein, 1)
#         print(food,'(',snapper_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
  
#     if food == 'tofu':
#         tofu_weight = vol*tofu_density
#         tofu_weight = round(tofu_weight, 2)
#         ### 除100
#         tofu_weight1 = tofu_weight/100
#         Calories = tofu_weight1*23
#         Calories = round(Calories, 1)
#         fat = tofu_weight1*0.1 ##脂肪
#         fat = round(fat, 1)
#         carbohydrate = tofu_weight1*4.5
#         carbohydrate = round(carbohydrate, 1)
#         protein = tofu_weight1*1.8
#         protein = round(protein, 1)
#         print(food,'(',tofu_weight,'g )','\n','========= Nutrition Facts =========','\n','總卡路里   碳水化合物  蛋白質  脂肪','\n', Calories,'kcal'  ,carbohydrate ,'g','    ',       protein,'g',' ',  fat,'g')
 
# #######################################################################################################
# ## 補零
# img_gray0 = cv2.imread('/home/kevin/yolov5work/yolov5/data/images/0.png', cv2.IMREAD_GRAYSCALE)

# c , d  = img_gray0.shape
# data2=[]##補零用
# data1=[]##存分割後圖片座標

# for i in range(c):
#     for j in range(d):
#         data2.append((i,j))
# original_depth0 = np.load("/home/kevin/yolov5work/yolov5/data/images/0.npy")##原始
# original_depth1 = np.load("/home/kevin/yolov5work/yolov5/data/images/1.npy")##後來

# for i in range(len(data2)):
        
    
#     original_depth01 = original_depth0[data2[i]]
#     original_depth01 = float(original_depth01)
#     original_depth11 = original_depth1[data2[i]]
#     original_depth11 = float(original_depth11)         
#     # print(i,truth_depth1)
    
#     if original_depth01 == 0.0:
#         original_depth0[data2[i]] = original_depth0[data2[i-1]]
#     if original_depth11 == 0.0:
#         original_depth1[data2[i]] = original_depth1[data2[i-1]]
# ## 補零
# #######################################################################################################
# total = []## 深度差
# local = []## 有深度差的座標
# local_depth = []## 座標原始深度

# # print(len(object_name_list))
# # print(coordinate)
# coordinate_array = np.array(coordinate)
# # print(coordinate_array[0])

# # for i in range(len(coordinate_array)):
# #     for j in range(4):
# #         print(coordinate_array[i][j])
# for g in range(len(object_name_list)):
#     ##  找出座標點
#     x1, y1, x2, y2 = coordinate_array[g][0],coordinate_array[g][1],coordinate_array[g][2],coordinate_array[g][3]
#     cropped_img = img_gray0[y1:y2, x1:x2]
#     a , b = cropped_img.shape
#     # print(cropped_img.shape)
#     for i in range(a):
#         for j in range(b):
#             data1.append((i,j))
        
#     for i in range(len(data1)):
#         # print(len(data1),i)
        
#         later_depth0 = original_depth0[ y1 : y2, x1: x2 ]##切割
#         later_depth1 = original_depth1[ y1 : y2, x1: x2 ]##切割

#         truth_depth1 = f'{later_depth1[data1[i]]}'
#         truth_depth0 = f'{later_depth0[data1[i]]}'
#         truth_depth1 = float(truth_depth1)
#         truth_depth0 = float(truth_depth0)
        
#         if (truth_depth1-truth_depth0) > 0 :##深度大於某數在計算 確保誤差
            
#             total.append(truth_depth1-truth_depth0)##深度差集合
#             f'{local.append(data1[i])}'##有深度差的座標集合
#     for i in range(len(total)):###有深度差大於某數的次數 也為像素數
        
#         # print('第二次',len(total),i,local[i])
#         truth_depth0 = f'{original_depth0[local[i]]}'###有深度差的座標原始深度  
#         local_depth.append(truth_depth0)#座標原始深度集合
    
#     local_depth[:] = [float(x) for x in local_depth]##https://whhnote.blogspot.com/2010/12/python-list.html
#     np_local_depth = np.array(local_depth)
#     ##https://ithelp.ithome.com.tw/articles/10226199
#     np_local_depth = np_local_depth/10
#     length = 0.00166187 * np_local_depth + 0.001747800933376259
#     area = length * length
#     # print(area)
#     total[:] = [float(x) for x in total]
#     np_total = np.array(total)
#     np_total = np_total/10
#     # print(area.shape)
#     # print(np_total.shape)
#     # print('前',len(data1))

#     vol = area * np_total## 體積
#     volume(object_name_list[g],sum(vol))
#     # print(object_name_list[g],'vol = ',sum(vol))
#     ### 將list清空
#     data1.clear()
#     total.clear()
#     local.clear()
#     local_depth.clear()
#     # print('後',len(data1))

        
    