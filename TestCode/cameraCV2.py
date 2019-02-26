# import the necessary packages
import numpy as np
import argparse
import cv2 as cv
import picamera
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)



def findcircles():
	# load the image, clone it for output, and then convert it to grayscale
	image = cv.imread("temp.jpg")
	output = image.copy()
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	# detect circles in the image
	circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 50)
	no_of_circles = 0  
	# ensure at least some circles were found
	if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		no_of_circles = len(circles)
	# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
			cv.circle(output, (x, y), r, (0, 255, 0), 4)
			cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	cv.imwrite('output.jpg',output)
	print( 'no of circles',no_of_circles)
	cv.waitKey(0)
	return


def captureimage():
	GPIO.setup(21,GPIO.OUT)
	camera = picamera.PiCamera()
	print( "relay on")
	GPIO.output(21,GPIO.LOW)
	time.sleep(1)
	camera.capture("temp.jpg")
	time.sleep(2)
	print( "relay off")
	GPIO.output(21,GPIO.HIGH)
	return

#main




captureimage()
findcircles()
