#############################
#IMPORT REQUIRED LIBRARIES:
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse
import cv2
import imutils
from collections import deque

def colourTracker(frame): 
    #############################
    #COLOUR TRACKING SETUP:
    #Define the lower and upper boundaries of the tracking colour "yellow"
    colourLower = (22, 150, 55) #HSV colour bound
    colourUpper = (40, 255, 255) #HSV colour bound
    pts = deque(maxlen=64) #Initialize the list of tracked points
    #pts = deque(maxlen=args["buffer"])

    image = imutils.resize(frame, width=600)       #Reduce size of frame
    blurred = cv2.GaussianBlur(image, (11, 11), 0) #Blur to reduce high freq noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #Convert to HSV
    
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colourLower, colourUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the mask and initialize the current
    # (x, y) center of the target
    cnts_col = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_col = imutils.grab_contours(cnts_col)
    center = None

    #############################
    #COLOUR TRACKING:
    # only proceed if at least one contour was found
    if len(cnts_col) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts_col, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        #Determine the distance to the objects centroid
        if center == None:
            dist = 0.00
        else:
            #dist = depth_frame.get_distance(center[0],center[1])
            dist = 10
        
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
        print("Centre is (COL): ", center)
        print("Distance to centre is (COL): ", dist)
        depth = dist #Store the dist and return to caller   
        #return x,y,depth 
        print(x,y,depth)  
        
    # update the points queue
    pts.appendleft(center)
    
    # else: 
        # print("NO TARGET FOUND!")
        # x,y,depth = -1,-1,-1
        # #return x,y,depth
        # print(x,y,depth)
        
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(image, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    #############################
    #CONTOUR GRABBING ON COLOUR MASK:
    ret,thresh = cv2.threshold(mask, 127, 255, 0)
     
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    color =  cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # print(hierarchy)
    # color =    cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
    mask = cv2.drawContours(color, contours,    -1, (0,255,0),    2)
    cv2.imshow("Contours",	color)
    cv2.imshow("thresh", thresh)

    return image