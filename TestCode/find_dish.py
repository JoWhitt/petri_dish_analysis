import cv2
import matplotlib
import numpy as np

img = cv2.imread('dish.jpg',0)
output = img.copy()
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100)
                    
					
if circles is not None:
	#circles = np.uint16(np.around(circles))
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([img, output]))
	cv2.waitKey(0)