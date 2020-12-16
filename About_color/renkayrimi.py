from __future__ import division
import cv2
import numpy as np
import sys
import os
import time
from numpy import array
 
def nothing(*arg):
        pass
 
# Initial HSV GUI slider values to load on program start.
#icol = (36, 202, 59, 71, 255, 255)    # Green
#icol = (18, 0, 196, 36, 255, 255)  # Yellow
#icol = (89, 0, 0, 125, 255, 255)  # Blue
#icol = (0, 100, 80, 10, 255, 255)   # Red
icol = (0, 0, 99, 205, 46, 255)  # White
cv2.namedWindow('colorTest')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)
 
# Raspberry pi file path example.
#frame = cv2.imread('/home/pi/python3/opencv/color-test/colour-circles-test.jpg')
# Windows file path example.
frame = cv2.imread('ball.jpg')
cam= cv2.VideoCapture(0)

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)
    
def colorResult(winname, img, x, y):
    cv2.resizeWindow(winname, 640,1000)
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)
    
while True:
    start_time = time.time() # start time of the loop
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')
    ret, frame=cam.read()
 
    # Show the original image.
    #cv2.imshow('frame', frame)
    
    # Blur methods available, comment or uncomment to try different blur methods.
    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
    #frameBGR = cv2.medianBlur(frameBGR, 7)
    #frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
    """kernal = np.ones((15, 15), np.float32)/255
    frameBGR = cv2.filter2D(frameBGR, -1, kernal)"""
	
    # Show blurred image.
    #cv2.imshow('blurred', frameBGR)
	
    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    
    # HSV values to define a colour range.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
    #cv2.imshow('mask-plain', mask)
    showInMovedWindow('Orjinal Goruntu',frame, 640, 0)
 
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    '''
    # Dimensions of the image
    sizeX = mask.shape[1]
    sizeY = mask.shape[0]

    #print(sizeX)
    #print(sizeY)
    startpoint_y = 0 
    endpoint_y = 640
    startpoint_x = [80,160,240,320,400,480,560,640]
    endpoint_x = [80,160,240,320,400,480,560,640]
    
  
    # Green color in BGR 
    color = (0, 255, 0) 
  
    # Line thickness of 1 px 
    thickness = 1
  
    # Using cv2.line() method,Ekranı eşit parçalara bölmek 
    for i in range(0,len(startpoint_x)):
        start_point=(startpoint_x[i],startpoint_y)
        end_point=(endpoint_x[i],endpoint_y)
        mask = cv2.line(mask, start_point, end_point, color, thickness) 
    '''
    # Show morphological transformation mask
    #cv2.imshow('mask', mask)
    showInMovedWindow('Maskelenmis Goruntu',mask, 640, 480)
    
    # Put mask over top of the original image.
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    # Show final output image
    #cv2.imshow('colorTest', result)
    colorResult('colorTest',result, 0, 0)


    #Siyah nokta sayacı
    th, threshed = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # filter by area 
    s1 = 0
    s2 = 7000000
    xcnts = [] 
    for cnt in cnts: 
       if s1<cv2.contourArea(cnt) <s2: 
          xcnts.append(cnt)
          
    print(len(xcnts),"cisim tespit edildi.") 
    #print("xcnts[0]-cisim1",xcnts[0])
    #print("xcnts[0]-cisim2",xcnts[1])

    for i in range(0,len(xcnts)):
        c = max(xcnts[i], key=cv2.contourArea)
        #print("c[1]",c[0][1])
        #print("agam",c[0][0]/80)
        if 230<c[0][1]<250:
            print(int(c[0][0]/80)+1,". kanaldan cisim geliyor")
            #print(c[0][0])





    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    time.sleep(0.5)

    #Drawimage
    sizeX = mask.shape[1]
    sizeY = mask.shape[0]

    #print(sizeX)
    #print(sizeY)
    startpoint_y = 0 
    endpoint_y = 640
    startpoint_x = [80,160,240,320,400,480,560,640]
    endpoint_x = [80,160,240,320,400,480,560,640]
    
  
    # Green color in BGR 
    color = (0, 255, 0) 
  
    # Line thickness of 1 px 
    thickness = 1
  
    # Using cv2.line() method,Ekranı eşit parçalara bölmek 
    for i in range(0,len(startpoint_x)):
        start_point=(startpoint_x[i],startpoint_y)
        end_point=(endpoint_x[i],endpoint_y)
        frame = cv2.line(frame, start_point, end_point, color, thickness)
        
    frame=cv2.line(frame, (0,230), (640,230), (0,0,255), thickness)
    frame=cv2.line(frame, (0,250), (640,250), (0,0,255), thickness)
    #cv2.imshow('image', frame)
    showInMovedWindow('Pozisyon Ayirici',frame, 1280, 0)
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop


    
cv2.destroyAllWindows()

