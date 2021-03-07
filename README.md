# Fiducial Marker and Colour Tracker

Python program to track object colour as well as Aruco Makers[1].

## Dependencies
* Python 3.x
* Numpy
* OpenCV 4.0+ 
* OpenCV 4.0+ Contrib modules

## Setup
pip install imutils
pip install opencv-python
pip install opencv-contrib-python

## Scripts
1. **tracker.py** : Calls the colour tracking script as well Aruco tracker library. Also detects the centroid of the aruco marker.

2. **arucoTracker.py**  : Based on Aruco Tracker script [1], used to calibrate the camera.

3. **colourTracker.py**  : Performs colour tracking and also outputs the colour mask for the detected colour.

## References
1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html


 
 
 
 
