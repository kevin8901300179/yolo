import numpy as np
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt

crop0 = cv2.imread("crop1.jpg")

a,b,c = crop0.shape

print(crop0.shape)
print(a*b)

cv2.imshow('My Image', crop0)

plt.show()
cv2.waitKey()

