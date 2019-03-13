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
horrible_idea = False

def button_callback(channel):
	time.sleep(0.5)
	global horrible_idea
	horrible_idea = True
	print("Button was pushed!")

def heater():
	humidity, temperature = Adafruit_DHT.read_retry(11,21)
	print(temperature)
	global horrible_idea
	if not horrible_idea:
		mylcd.lcd_clear()
		mylcd.lcd_display_string("   IncuBetter",1)
		mylcd.lcd_display_string("Temp: %s" %temperature,2)
		if(temperature <=36):
			GPIO.setup(4,GPIO.OUT) #heater relay
			GPIO.output(4,GPIO.LOW)



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
	image_subtraction_approach()


def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return resized

def get_dish_mask(image, circle):
	(x, y, r) = circle
	mask = np.zeros(image.shape, np.uint8) # initialise image
	cv.circle(mask, (x, y), int(r/2), (255,255,255), r+1) # draw circle into mask
	return mask


def count_keypoints(keypoints):
    count = 0
    for kp in keypoints:
        count+=1
    return count


def find_blobs(img):
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100;
    params.maxThreshold = 5000;

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 200

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.9

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.95

    #Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
	count = count_keypoints(keypoints)
    print(count)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img2 = img.copy()
    for marker in keypoints:
        img2 = cv.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(255, 0, 0))
    cv.imshow("output",img2)
    cv.waitKey(0)




def image_subtraction_approach():
	global pic_num
    filename_0 = 'picture0.jpg'
    filename_1 = 'picture1.jpg'
    scale_factor = 0.25 # this is the scale to which images are resized for screen display


    # read in images (for before & after growth)
    empty_dish = cv.imread(filename_0)
    full_dish = cv.imread(filename_1)
    assert (empty_dish is not None)
    assert (full_dish is not None)

    # get subtraction image
    sub_result = cv.subtract(empty_dish, full_dish)
    display_image = np.hstack((empty_dish, full_dish, sub_result))

    # convert to grayscale, threshold & erode image
    binary_threshold = 100
    sub_result = cv.cvtColor(sub_result, cv.COLOR_BGR2GRAY)
    ret, thresholded = cv.threshold(sub_result, binary_threshold, 255, 0)

    kernel = np.ones((15,15),np.uint8)
    thresholded = cv.erode(thresholded, kernel)
    thresholded = cv.dilate(thresholded, kernel)


    # get image contours
    im2, contours, hierarchy = cv.findContours(thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # analyse each contour region
    # & remove any contours which don't represent a bacteria colony
    new_contours = []
    result_image = full_dish.copy()
    for c in contours:
        mask = np.zeros(thresholded.shape,np.uint8)
        cv.drawContours(mask,[c],0,255,-1)
        mean_pixel_val = cv.mean(thresholded, mask=mask)
        # mean val shows whether the region is a colony, or a hole in a detected colony
        # holes will be dark, whereas colonies will be light

        if int(mean_pixel_val[0]) > 200: # region is a colony, not just a hole

            # analyse contour shape, to separate overlapping contours
            hull = cv.convexHull(c)
            # cv.drawContours(result_image, [hull], 0, (0,255,0))

            # if area_between_contour_&_hull > threshold:  <----+
            #   get region of interest (sub_image)              |
            #   num_of_contours = get_contours(sub_image)       |
            #   while (num_of_contours == 1):                   |
            #       sub_image = erode(sub_image)                |
            #       num_of_contours = get_contours(sub_image)   |
            # iterate back to here -----------------------------+

            new_contours.append(c)

    # create contour images
    cv.drawContours(result_image, new_contours, -1, (0,0,255), 1)

    # print results
    count = len(new_contours)
    print (count, "colonies found")
    print (len(contours)-count, "false posatives disregarded")

    # add text & show images
    thresholded = resize_image(thresholded, scale_factor)
    #cv.imshow("thresholded",thresholded)

    result_image = resize_image(result_image, scale_factor)
    cv.putText(result_image, str(count), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
#    cv.imshow("result_image",result_image)
	mylcd.lcd_clear()
	mylcd.lcd_display_string('colonies : ' + str(count),2)
	mylcd.lcd_clear()
	mylcd.lcd_display_string("   IncuBetter",1)
	pic_num = pic_num +1
	cv.waitKey(0)
    cv.destroyAllWindows()


def find_dishes():
    ''' load the image, clone it for output, and convert to grayscale '''
    image = cv.imread(image_path)
    assert(image is not None)  # image read successfully

    output = image.copy()

    gray = cv.medianBlur(image,5)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ''' apply hough algorithm to locate dish '''
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100)
    assert(circles is not None)  # dish found
    assert(len(circles)==1)  	 # only one dish found

    #circles = np.uint16(np.around(circles))
    circles = np.round(circles[0, :]).astype("int")

    # ''' draw circle into output image '''
    # (x, y, r) = circles[0]
    # cv.circle(output, (x, y), r, (0, 255, 0), 4)
    # cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    ''' create & apply mask '''
    mask = get_dish_mask(image, circles[0])
    output = cv.bitwise_and(output, mask)


    ''' code to find blobs '''
    img = output
    image_edged = cv.Canny(gray, 90, 100)
    image_edged = cv.dilate(image_edged, None, iterations=1)
    image_edged = cv.erode(image_edged, None, iterations=2)
    cv.imshow("output",image_edged)
    cv.waitKey(0)




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
	if horrible_idea:
		print("got here")
		captureimage()
		time.sleep(1)
		horrible_idea = False


GPIO.cleanup()
