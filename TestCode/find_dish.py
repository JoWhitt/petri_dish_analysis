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
    # read in images (for before & after growth)
    empty_dish = cv.imread('..\\sample_p_dish_images\\sample_empty.jpg')
    full_dish = cv.imread('..\\sample_p_dish_images\\sample_full.jpg')

    # get subtraction image & convert to grayscale
    sub_result = cv.subtract(empty_dish, full_dish)
    sub_result = cv.cvtColor(sub_result, cv.COLOR_BGR2GRAY)
    # display_image = np.hstack((empty_dish, full_dish, sub_result))

    # threshold & erode image
    ret, thresholded = cv.threshold(sub_result, 10, 255, 0)

    kernel = np.ones((5,5),np.uint8)
    thresholded = cv.erode(thresholded, kernel)

    # get image contours
    im2, contours, hierarchy = cv.findContours(thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # create contour image
    result_image = full_dish.copy()
    cv.drawContours(result_image, contours, -1, (0,255,0), 1)

    # count colonies & display on image
    count = len(contours)
    cv.putText(result_image, str(count), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))


    # cv.imshow("display_image",display_image)
    # cv.imshow("contour_image",contour_image)
    cv.imshow("result_image",result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# find_dishes()
# find_blobs(img)
image_subtraction_approach()


# show the output image
#cv.imshow("output", np.hstack([image, image_edged]))
#cv.waitKey(0)