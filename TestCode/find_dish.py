import cv2 as cv
import numpy as np


image_path = "green_dots.jpg"

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


''' code to find blobs '''
img = output
image_edged = cv.Canny(gray, 90, 100)
image_edged = cv.dilate(image_edged, None, iterations=1)
image_edged = cv.erode(image_edged, None, iterations=2)
cv.imshow("output",image_edged)
cv.waitKey(0)


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

find_blobs(img)

# show the output image
#cv.imshow("output", np.hstack([image, image_edged]))
#cv.waitKey(0)