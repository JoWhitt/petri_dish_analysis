import cv2 as cv
import numpy as np

image_path = "..\\sample_p_dish_images\\green_dots.jpg"

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def get_dish_mask(image, circle):
	(x, y, r) = circle

	mask = np.zeros(image.shape, np.uint8) # initialise image
	cv.circle(mask, (x, y), int(r/2), (255,255,255), r+1) # draw circle into mask

	return mask

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

# show the output image
cv.imshow("output", np.hstack([image, output]))
cv.waitKey(0)