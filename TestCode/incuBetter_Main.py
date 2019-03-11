# import the necessary packages
import numpy as np
import argparse
import cv2 as cv
import picamera
import time
import RPi.GPIO as GPIO
import sys
import Adafruit_DHT
import I2C_LCD_driver
import datetime

pic_num = 0
def button_callback(channel):

	time.sleep(0.5)
	print("Button was pushed!")
	captureimage()



def heater():
	humidity, temperature = Adafruit_DHT.read_retry(11,21)
	print(temperature)
	mylcd.lcd_display_string("Temp: %s" %temperature,2)
	if(temperature <=37):
		GPIO.setup(4,GPIO.OUT) #heater relay
		GPIO.output(4,GPIO.LOW)


def findcircles():
	global pic_num
	# load the image, clone it for output, and then convert it to grayscale
	image = cv.imread("picture" + str(pic_num) + ".jpg")
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
	cv.imwrite("output" + str(pic_num) + ".jpg",output)
	mylcd.lcd_clear()
	print( 'no of circles',no_of_circles)
	mylcd.lcd_display_string('circles : ' + str(no_of_circles),2)
	mylcd.lcd_clear()
	mylcd.lcd_display_string("   IncuBetter",1)
	cv.waitKey(0)
	pic_num = pic_num +1
	return


def captureimage():
	camera = picamera.PiCamera() #camera
	camera.brightness = 50
	camera.shutter_speed = 5500000000000
	GPIO.output(17,GPIO.LOW)
	mylcd.lcd_clear()
	mylcd.lcd_display_string("Flash on",2)
	time.sleep(2)
	camera.capture("picture" + str(pic_num) + ".jpg")
	print("picture taken")
	mylcd.lcd_clear()
	mylcd.lcd_display_string("Picture Taken",2)
	time.sleep(2)
	GPIO.output(17,GPIO.HIGH)
	mylcd.lcd_clear()
	mylcd.lcd_display_string("Flash off",2)
	camera.close()
	findcircles()


	return





GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(16,GPIO.IN, pull_up_down=GPIO.PUD_UP) #button
GPIO.setup(4,GPIO.OUT) #heater relay
GPIO.setup(17,GPIO.OUT) #flash relay
GPIO.output(17,GPIO.HIGH)

mylcd=I2C_LCD_driver.lcd()
mylcd.lcd_display_string("   IncuBetter",1)

INTERVAL = datetime.timedelta(hours=6)
GPIO.add_event_detect(16,GPIO.FALLING,callback=button_callback,bouncetime=10000) # Setup event on pin 16 falling edge
prev_picture = datetime.datetime.now()



while True:
	heater()
	if datetime.datetime.now() > prev_picture + INTERVAL:
		prev_picture = datetime.datetime.now()
		captureimage()
		findcircles()


GPIO.cleanup()

