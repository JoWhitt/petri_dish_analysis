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
os.nice(20)

mylcd=I2C_LCD_driver.lcd()



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
    print("Temp is {}".format(Temp))
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Temp: %s" %Temp,2)

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


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(20,GPIO.IN, pull_up_down=GPIO.PUD_UP) #up
GPIO.setup(16,GPIO.IN, pull_up_down=GPIO.PUD_UP) #down
GPIO.setup(8,GPIO.IN, pull_up_down=GPIO.PUD_UP) #next
GPIO.setup(7,GPIO.IN, pull_up_down=GPIO.PUD_UP) #back

GPIO.add_event_detect(20,GPIO.FALLING,callback=up_btn,bouncetime=100) #
GPIO.add_event_detect(16,GPIO.FALLING,callback=down_btn,bouncetime=100) #
GPIO.add_event_detect(8,GPIO.FALLING,callback=next_btn,bouncetime=100) #
GPIO.add_event_detect(7,GPIO.FALLING,callback=back_btn,bouncetime=100) #

main()
