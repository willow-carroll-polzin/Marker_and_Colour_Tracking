
#############################
#IMPORT REQUIRED LIBRARIES:
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import cv2.aruco as aruco
import glob
import imutils
import time
import math as mt
import statistics as st

from arucoTracker import arucoTrackerCal
from colourTracker import colourTracker

#############################
#PROGRAM SETUP:
#Select tracking types - ENABLE a tracking method by setting to TRUE
COLOUR_STATUS = False
ARUCO_STATUS = False

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

#If a video path was not supplied, grab the reference to the local webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
#Otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

#Calibrate aurco detector
mtx, dist = arucoTrackerCal()

#Look for the aruco code
while True:
    #############################
    #INPUT SETUP:
    #Grab the current frame
    frame = vs.read()
    
    #Handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    font = cv2.FONT_HERSHEY_COMPLEX 
    
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        print("NO FRAMES")
        break
    
    #############################
    #COLOUR TRACKING: 
    if COLOUR_STATUS == True:

        image = colourTracker(frame)

        # show the frame to our screen
        cv2.imshow("Colour", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    #############################
    #ARUCO TRACKING:  
    if ARUCO_STATUS == True:
        #Select dictionary based on the aruco marker being used
        #aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        #aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)
        #aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

        #Gray scale the input frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detector parameters can be set here
        #List of detection parameters: https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html)
        parameters =  aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print("ARUCO CORNERS: ", corners)

        #Store marker centroid data
        dist_thresh = 1000
        centroids = []

        #Ensure that the id list is not empty
        if np.all(ids != None):
            print("ARUCO DETECTED!")
            #Estimate pose of each marker and return the values
            #rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                #Daw axis for the aruco markers
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            #Draw a square around the markers
            aruco.drawDetectedMarkers(frame, corners)

            #Write the ids for each marker detected
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '
            cv2.putText(frame, "Id: " + strg, (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

            #Find centroid of aruco code
            for i in range(len(corners[0][0])):
                for j in range(i+1,len(corners[0][0])):
                    val = mt.sqrt((corners[0][0][i][0]-corners[0][0][j][0])**2+(corners[0][0][i][1]-corners[0][0][j][1])**2)
                    if val < dist_thresh: 
                        centroids.append(corners[0][0][i])
                        centroids.append(corners[0][0][j])
            
            #Remove duplicates from list:
            for i in range(len(centroids)):
                centroids[i]=tuple(centroids[i])
            centroids=list(set(centroids))  
            
            avg_centroid_X = []
            avg_centroid_Y = []
            centroid_X = [] 
            centroid_Y = []
            for k in range(len(centroids)):
                centroid_X.append(centroids[k][0])
                centroid_Y.append(centroids[k][1])
            
            #Only average if data recieved, else set centroid average to arb value
            if (len(centroid_X) == 0 or len(centroid_Y) == 0):
                avg_centroid_X = 0
                avg_centroid_Y = 0
            else: 
                avg_centroid_X = st.mean(centroid_X)
                avg_centroid_Y = st.mean(centroid_Y)
            avg_centroid = [avg_centroid_X, avg_centroid_Y]
            print ("Average of cnt's is", avg_centroid)
            
            #Determine the distance to the shapes centroid (if depth camera is used)
            # if avg_centroid == None:
            #     dist = 0.00
            # else:
            #     #dist = depth_frame.get_distance(avg_centroid[0],avg_centroid[1])    
            # depth = dist #Store the dist and return to caller
            
        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        
        # display the resulting frame
        cv2.imshow('TESTER 1',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
    
vs.release() #Release video stream
cv2.destroyAllWindows() #Close all windows