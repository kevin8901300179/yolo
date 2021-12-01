import os
import shutil
import time

pathTest = "/home/kevin/yolov5work/yolov5/data/images/"


try:
    shutil.rmtree(pathTest)## 刪除資料夾
except OSError as e:
    print(e)
else:
    print("The directory is deleted successfully")
    os.mkdir('images')##馬上在新增資料夾
