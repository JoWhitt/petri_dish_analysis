# import the necessary packages
import numpy as np
import argparse
import cv2 as cv
import picamera
import time
import RPi.GPIO as GPIO
import sys
import Adafruit_DHT
import datetime

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(16,GPIO.IN, pull_up_down=GPIO.PUD_UP) #button
GPIO.setup(4,GPIO.OUT) #heater relay
GPIO.setup(17,GPIO.OUT) #flash relay
GPIO.output(17,GPIO.HIGH)


INTERVAL = datetime.timedelta(hours=6)

def heater():
	humidity, temperature = Adafruit_DHT.read_retry(11,26)
	print(temperature)
	if(temperature <=41):
		GPIO.setup(4,GPIO.OUT) #heater relay
		GPIO.output(4,GPIO.LOW)


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
	camera = picamera.PiCamera() #camera
	GPIO.output(17,GPIO.LOW)
	time.sleep(1)
	camera.capture("temp.jpg")
	print("picture taken")
	time.sleep(2)
	GPIO.output(17,GPIO.HIGH)
	camera.close()
	return

#main


prev_picture = datetime.datetime.now()

while True:
	heater()
	if datetime.datetime.now() > prev_picture + INTERVAL:
		prev_picture = datetime.datetime.now()
		captureimage()
		findcircles()



#while True:
#	heater()
#	try:
#		GPIO.wait_for_edge(16, GPIO.FALLING)
#		time.sleep(2)
#		captureimage()
#		findcircles()
#	except KeyboardInterrupt:
#		GPIO.cleanup()
#GPIO.cleanup()


