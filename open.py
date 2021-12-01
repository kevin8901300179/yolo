import numpy as np
import cv2

# 建立一張 512x512 的 RGB 圖片（黑色）
img = np.zeros((256, 256, 3), np.uint8)

text ='總卡路里'
cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
1, (0, 255, 255), 1, cv2.LINE_AA)

# 在圖片上畫一條紅色的對角線，寬度為 5 px
cv2.line(img, (150, 0), (150, 255), (0, 0, 255), 2)

# 顯示圖片
cv2.imshow('My Image', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()