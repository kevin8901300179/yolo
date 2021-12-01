import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

source_file_name = '{}.jpg'.format(time.strftime("%Y-%m-%d %H:%M"))
# i = 'foodimg{counter:03d} time{timestamp:%Y-%m-%d-%H-%M}.jpg'
cv2.imwrite(source_file_name, frame)
print(source_file_name)
#photo
time.sleep(2)

