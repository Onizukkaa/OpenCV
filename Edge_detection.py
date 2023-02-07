# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:55:33 2022

@author: joachim
"""

"""EDGE DETECTION"""
#%%

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#%%

img=cv.imread("panda.jpg")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)


#Laplacian
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))

#◘Sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel=cv.bitwise_or(sobelx,sobely)

canny=cv.Canny(gray,150,175)

cv.imshow("Output",canny)
cv.waitKey(0)

#plt.imshow(img,cmap="gray")