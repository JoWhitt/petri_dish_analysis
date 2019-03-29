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
import os
from enum import Enum
import threading
os.nice(20)

mylcd=I2C_LCD_driver.lcd()
pic_num = 0
currentTemp = 0




def up_btn(channel):            #button interupt
    print("up was pushed!")
    if Current_state == States.SET_TEMP:
        global Temp
        Temp +=1

    elif Current_state == States.SET_TIME:
        global Time
        Time += 1

    elif Current_state == States.SET_INTERVAL:
        global Time_interval
        Time_interval +=1

def down_btn(channel):           #button interupt
    print("down was pushed!")
    if Current_state == States.SET_TEMP:
        global Temp
        Temp -=1

    elif Current_state == States.SET_TIME:
        global Time
        Time -= 1

    elif Current_state == States.SET_INTERVAL:
        global Time_interval
        Time_interval -=1

    elif Current_state == States.DISPLAY_MODE:
        captureimage()


def next_btn(channel):           #button interupt
    print("next was pushed!")
    global Current_state
    if Current_state != States.TAKING_PIC and Current_state != States.DISPLAY_MODE and Current_state != States.PREHEATING:
        Current_state = Current_state.succ()
    elif Current_state == States.LOAD_DISHES:
        Current_state = States.TAKING_PIC

def back_btn(channel):           #button interupt
    print("back was pushed!")
    global Current_state
    if Current_state == States.SET_TIME or Current_state == States.SET_INTERVAL:
        Current_state = Current_state.prev()




class States(Enum):
    SET_TEMP = 0
    SET_TIME = 1
    SET_INTERVAL = 2
    PREHEATING = 3
    LOAD_DISHES = 4
    TAKING_PIC = 5
    DISPLAY_MODE = 6


    def succ(self):
        v = self.value + 1
        if v > 6:
            return States(self.value)
        return States(v)

    def prev(self):
        v = self.value - 1
        if v < 0:
            return States(self.value)
        return States(v)

class Actions(Enum):
    INC = 0
    DECREASE = 1
    NEXT = 2
    PREV = 3

Current_state = States.SET_TEMP


Temp = 37
Time = 24.00 # hours
Time_interval = 6


def display_temp():
    print("Temp is {}".format(Temp))
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Temp: %s" %Temp,2)

def display_current_temp():
    global currentTemp
    print("cur-Temp is {}".format(currentTemp))
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Temp: %s" %currentTemp,2)

def display_time():
    print("Time is {}".format(Time))
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Time: %s" %Time,2)

def display_interval():
    print("Time interval is {}".format(Time_interval))
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Time int: %s" %Time_interval,2)

def update():
    global Current_state
    Current_state = Current_state.succ()



def take_pic():
    pass

def display_stuff():
    print("Display")
    mylcd.lcd_clear()
    mylcd.lcd_display_string("displaying stuff",2)

    pass


def main():

    while True:

        if Current_state == States.SET_TEMP:

            display_temp()

        elif Current_state == States.SET_TIME:
            thread_1 = threading.Thread(target=heater, args=())
            thread_1.start()

            display_time()

        elif Current_state == States.SET_INTERVAL:

            display_interval()

        elif Current_state == States.PREHEATING:

            display_current_temp()
            update()

            # just needs to check if current temp is set temp than changes state

        elif Current_state == States.LOAD_DISHES:

            print("Load dishes")
            mylcd.lcd_display_string("load Dishes",2)

            pass

        elif Current_state == States.TAKING_PIC:

            print("take pic")
           # mylcd.lcd_display_string("take pic",2)
            captureimage()

            # just needs changes state after picture

            update()

        elif Current_state == States.DISPLAY_MODE:

            display_stuff()

        # This was added for testing purposes

        time.sleep(1)





def captureimage():
    global pic_num
    GPIO.setup(17,GPIO.OUT)
    camera = picamera.PiCamera() #camera
    camera.brightness = 50
    camera.shutter_speed = 5500000000000
    GPIO.output(17,GPIO.LOW)
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Flash on",2)
    time.sleep(2)
    camera.capture("/home/pi/transfers/picture" + str(pic_num) + ".jpg")
    print("picture taken")
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Picture Taken",2)
    time.sleep(2)
    GPIO.output(17,GPIO.HIGH)
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Flash off",2)
    time.sleep(1)
    mylcd.lcd_clear()
    camera.close()
    if(pic_num != 0):
        mylcd.lcd_display_string("  Analyzing",1)
        mylcd.lcd_display_string("   Images...",2)
        count_colonies()
    pic_num = pic_num + 1


def heater():
    while True:
        global currentTemp
        humidity, currentTemp = Adafruit_DHT.read_retry(11,21)		#get reading from temperature sensor
        print(currentTemp)
        if(currentTemp <= Temp):
            GPIO.setup(4,GPIO.OUT)		 #heater relay setup
            GPIO.output(4,GPIO.LOW)		 #heater on
        else: currentTemp = Temp


def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return resized

def image_subtraction_approach(empty_dish, full_dish):
    scale_factor = 1.3# 0.25 # this is the scale to which images are resized for screen display

    # get subtraction image
    sub_result = cv.subtract(empty_dish, full_dish)
    display_image = np.hstack((empty_dish, full_dish, sub_result))

    # convert to grayscale, threshold & erode image
    binary_threshold = 100
    sub_result = cv.cvtColor(sub_result, cv.COLOR_BGR2GRAY)
    ret, thresholded = cv.threshold(sub_result, binary_threshold, 255, 0)

    kernel = np.ones((5,5),np.uint8)
    thresholded = cv.erode(thresholded, kernel)
    thresholded = cv.dilate(thresholded, kernel)


    # get image contours
    contours, hierarchy = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
            new_contours.append(c)

            # analyse contour shape, to separate overlapping contours
            hull = cv.convexHull(c)
            area = cv.contourArea(c)
            hull_area = cv.contourArea(hull)
            solidity = float(area)/hull_area
            print(solidity)
            if solidity < 0.9:
                new_contours.remove(c)
                x,y,w,h = cv.boundingRect(c)
                roi = thresholded[y:y+h,x:x+w]
                cv.rectangle(result_image,(x,y),(x+w,y+h),(0,0,255),2)
                contours_2, hierarchy = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                while len(contours_2) == 1:
                    kernel = np.ones((5,5),np.uint8)
                    roi = cv.erode(roi, kernel)
                    #cv.imshow('eroded image', roi)
                    contours_2, hierarchy = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours_2:
                    new_contours.append(cnt)
            # cv.drawContours(result_image, [hull], 0, (0,255,0))

            # if area_between_contour_&_hull > threshold:  <----+
            #   get region of interest (sub_image)              |
            #   num_of_contours = get_contours(sub_image)       |
            #   while (num_of_contours == 1):                   |
            #       sub_image = erode(sub_image)                |
            #       num_of_contours = get_contours(sub_image)   |
            # iterate back to here -----------------------------+

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
    #cv.putText(result_image, str(count), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
    #cv.imshow("result_image",result_image)

    #cv.waitKey(0)
    #cv.destroyAllWindows()


''' read in two images (containing empty and filled petri dishes respectively)
display & return two cropped images of the same scale - containing the two dishes '''
def get_cropped_image(input_image):
    ''' 1. FIND DISH '''
    # smooth image & convert to grayscale
    smoothed = cv.medianBlur(input_image,5)
    gray = cv.cvtColor(smoothed, cv.COLOR_BGR2GRAY)

    # apply hough algorithm to locate dish
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100)
    assert (circles is not None) and (len(circles)==1)  # assert one and only one dish found

    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]
    ''' 2. CREATE & APPLY MASK '''
    mask = np.zeros(input_image.shape, np.uint8) # initialise image
    cv.circle(mask, (x, y), int(r/2), (255,255,255), r+1) # draw circle into mask
    masked_image = cv.bitwise_and(input_image, mask)

    ''' 3. CROP IMAGE '''
    cropped = masked_image[(y-r):(y+r), (x-r):(x+r)]

    return cropped

def split_image(image):
    height, width = image.shape[:2]

    # Top left
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .5), int(width* .5)
    top_left = image[start_row:end_row , start_col:end_col]

    # Top right
    start_row, start_col = int(0), int(width * .45)
    end_row, end_col = int(height * .55), int(width)
    top_right = image[start_row:end_row , start_col:end_col]

    # Bottom left
    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width* .5)
    bottom_left = image[start_row:end_row , start_col:end_col]

    # bottom right
    start_row, start_col = int(height * .5), int(width * .5)
    end_row, end_col = int(height), int(width)
    bottom_right = image[start_row:end_row , start_col:end_col]

    top_left.size
    top_right.size
    bottom_left.size
    bottom_right.size

    images = [top_left,top_right,bottom_left,bottom_right]

    return images



def count_colonies():
    global pic_num
    empty_image = cv.imread("/home/pi/transfers/picture0.jpg")
    full_image = cv.imread("/home/pi/transfers/picture" + str(pic_num) + ".jpg")

    split_empty_image = split_image(empty_image)
    split_full_image = split_image(full_image)

    zip_image = zip(split_empty_image,split_full_image)
    colony_count = []

    for i,j in zip_image:
        #cv.imshow("image",j)
        #cv.waitKey(0)

        assert (i is not None) and (j is not None)
        empty_cropped = get_cropped_image(i)
        full_cropped = get_cropped_image(j)

    # scale images to match
        if empty_cropped.shape[0] != full_cropped.shape[0]: # images are different sizes
            scale_factor = full_cropped.shape[0]/empty_cropped.shape[0]
            empty_cropped = resize_image(empty_cropped, scale_factor)

        colony_count.append( image_subtraction_approach(empty_cropped, full_cropped))

    print(colony_count)



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(20,GPIO.IN, pull_up_down=GPIO.PUD_UP) #up
GPIO.setup(16,GPIO.IN, pull_up_down=GPIO.PUD_UP) #down
GPIO.setup(8,GPIO.IN, pull_up_down=GPIO.PUD_UP) #next
GPIO.setup(7,GPIO.IN, pull_up_down=GPIO.PUD_UP) #back

GPIO.add_event_detect(20,GPIO.BOTH,callback=up_btn,bouncetime=200) #
GPIO.add_event_detect(16,GPIO.BOTH,callback=down_btn,bouncetime=200) #
GPIO.add_event_detect(8,GPIO.BOTH,callback=next_btn,bouncetime=200) #
GPIO.add_event_detect(7,GPIO.BOTH,callback=back_btn,bouncetime=200) #

main()
