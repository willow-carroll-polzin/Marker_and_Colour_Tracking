
import numpy as np
import cv2
import cv2.aruco as aruco
import glob

#cap = cv2.VideoCapture(1)

def arucoTrackerCal(): 
    ####---------------------- CALIBRATION ---------------------------
    #Termination criteria for the iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    #A checkerboard of size (7 x 6) is used
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    #Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #Iterating through all calibration images
    images = glob.glob('calib_images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Find the chess board (calibration pattern) corners
        ret, corners = cv2.findChessboardCorners(grey, (7,6),None)

        #If calibration pattern is found, add object points, image points
        if ret == True:
            objpoints.append(objp)

            #Refine the corners of the detected corners
            corners2 = cv2.cornerSubPix(grey,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            #Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grey.shape[::-1],None,None)
    return mtx, dist
    
def arucoTracker(frame, mtx, dist): 
    ###------------------ ARUCO TRACKER ---------------------------
    #Select dictionary based on the aruco marker being used
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

    #Gray scale the input frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detector parameters can be set here
    #List of detection parameters: https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html)
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    #Lists of ids and the corners belonging to each id (id read from marker)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grey, aruco_dict, parameters=parameters)
    print("ARUCO CORNERS: ", corners)

    #Font for displaying text on the frame
    #font = cv2.FONT_HERSHEY_SIMPLEX

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
    else:
        #Show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    #Display new frame
    cv2.imshow('Aruco Detector',frame)
    #key = cv2.waitKey(1) & 0xFF

    if (len(corners) > 0):
        return (ids, corners, True)
    else: 
        return (ids, corners, False)
    # References
    # 1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    # 2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    # 3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html