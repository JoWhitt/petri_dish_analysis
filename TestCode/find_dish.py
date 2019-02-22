import cv2 as cv
# import matplotlib
import numpy as np
import argparse

image_path = "..\\sample_p_dish_images\\green_dots.jpg"

''' load the image, clone it for output, and then convert it to grayscale '''
image = cv.imread(image_path)
assert(image is not None) 

output = image.copy()
image = cv.medianBlur(image,5)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cimg = cv.cvtColor(image,cv.COLOR_GRAY2BGR)

''' apply hough algorithm to locate dish '''
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100)
                    			
if circles is not None:
	#circles = np.uint16(np.around(circles))
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		#  circle
		cv.circle(output, (x, y), r, (0, 255, 0), 4)
		cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv.imshow("output", np.hstack([image, output]))
	cv.waitKey(0)