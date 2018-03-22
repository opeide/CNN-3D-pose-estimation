import numpy as np
import cv2
from batch_generator import BatchGenerator
import util

path="dataset/coarse/ape/coarse10.png"

loaded_img = cv2.imread(path)
cv2.imshow('pic_rt', loaded_img)
loaded_img_min = loaded_img.min(axis=(0, 1), keepdims=True)
loaded_img_max = loaded_img.max(axis=(0, 1), keepdims=True)
normalImg = (loaded_img - loaded_img_min)/(loaded_img_max - loaded_img_min)
print("normalized image", normalImg)
#print(loaded_img.shape)
print(normalImg.shape)
cv2.imshow('dst_rt', normalImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
