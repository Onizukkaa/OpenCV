# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:26:15 2022

@author: joachim
"""
"""OPENCV"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
"""IMAGE"""

img=cv2.imread("pandas_2.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#cv2.imshow("Output",img)
#cv2.waitKey(0)

#plt.imshow(img,cmap="gray")
#%%Convertir en gris

imgGray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray image",imgGray)
cv2.waitKey(0)
#%% Floue
imgBlur=cv2.GaussianBlur(imgGray,(3,3),0)
cv2.imshow("Blur image",imgBlur)
cv2.waitKey(0)
#%% Detecteur bord
imgCanny=cv2.Canny(img,100,100)
cv2.imshow("Canny image",imgCanny)
cv2.waitKey(0)
#%%Dilatation
kernel=np.ones((5,5),np.uint8)

imgDialation=cv2.dilate(imgCanny,kernel,iterations=1)
cv2.imshow("Dialation image",imgDialation)
cv2.waitKey(0)
#%% Erosion
kernel=np.ones((5,5),np.uint8)

imgEroded=cv2.erode(imgDialation,kernel,iterations=1)
cv2.imshow("Eroded image",imgEroded)
cv2.waitKey(0)
#%% Redimensionner
#print(img.shape) #768,1366,3
imgResize=cv2.resize(img,(500,800))
cv2.imshow("resize image",imgResize)
cv2.waitKey(0)
#print(imgResize.shape)#800,500,3
#%% Recadrer
imgCropped=img[0:600,200:500]
cv2.imshow("Cropped image",imgCropped)
cv2.waitKey(0)
#%%Warp prespective
img=cv2.imread("carte.jpeg")
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow(" image",img)
cv2.waitKey(0)

#plt.imshow(img,cmap="gray")
#%%
width,height=250,350

pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgOutput=cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow(" image",imgOutput)
cv2.waitKey(0)
#%%Joining images
img=cv2.imread("carte.jpeg")
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#%% 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
#%%
img=cv2.imread("pandas_2.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_stack=stackImages(0.5,([img,img_gray,img],[img,img,img ]))
cv2.imshow("image",img_stack)
cv2.waitKey(0)

#%% COLOR Detection

path="carte.jpeg"


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue min","TrackBars",0,179,empty )
cv2.createTrackbar("Hue max","TrackBars",13,179,empty )
cv2.createTrackbar("sat min","TrackBars",20,255,empty )
cv2.createTrackbar("sat max","TrackBars",255,255,empty )
cv2.createTrackbar("val min","TrackBars",174,255,empty )
cv2.createTrackbar("val max","TrackBars",255,255,empty )

while True: 
    img=cv2.imread(path)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)      
    h_min=cv2.getTrackbarPos("Hue min", "TrackBars")
    h_max=cv2.getTrackbarPos("Hue max", "TrackBars")
    s_min=cv2.getTrackbarPos("sat min", "TrackBars")
    s_max=cv2.getTrackbarPos("sat max", "TrackBars")
    v_min=cv2.getTrackbarPos("val min", "TrackBars")
    v_max=cv2.getTrackbarPos("val max", "TrackBars")
    
    print(h_min,h_max,s_min,s_max,v_min,v_max)  
    
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)
    img_result=cv2.bitwise_and(img,img,mask=mask)
     
    #cv2.imshow("image",imgHSV)
    #cv2.imshow("image",mask)
    #cv2.imshow("image",img_result)
    
    img_stack=stackImages(0.6,([img,imgHSV],[mask,img_result]))
    
    cv2.imshow("image",img_stack)
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break                 

#%% CONTOUR/SHAPE DETECTION
path="forme.jpg"

img=cv2.imread(path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img,cmap="gray")    
#%%
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: 
                objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: 
                    objectType= "Square"
                else:
                    objectType="Rectangle"
            elif objCor ==5:
                objectType="Pentagone"
            elif objCor>5: 
                objectType= "Circles"
            else:
                objectType="None"



            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)

imgContour=img.copy()
img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_blur=cv2.GaussianBlur(img_gray,(7,7),1)
img_canny=cv2.Canny(img_blur,50,50)
img_blank=np.zeros_like(img)

getContours(img_canny)

img_stack=stackImages(0.3,([img,img_gray,img_blur],[img_canny,imgContour,img_blank]))
cv2.imshow("image",img_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()            
                
#%% DETECTION FACIALE
path="visage.png"
img=cv2.imread(path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img,cmap="gray") 
#%% 

face_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
faces=face_cascade .detectMultiScale(img_gray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#%%
"""VIDEO"""

#cap=cv2.VideoCapture("test_video.mp4")

#for webcam
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100) #luminosité

i=0
j=1
while True:
    success,img=cap.read()
    cv2.imshow("Vidéo",img)
    i+=1
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break 
    
print(i)