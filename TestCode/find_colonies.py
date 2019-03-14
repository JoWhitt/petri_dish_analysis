import cv2 as cv
import numpy as np

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
    cv.imshow("thresholded",thresholded)

    result_image = resize_image(result_image, scale_factor)
    cv.putText(result_image, str(count), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
    cv.imshow("result_image",result_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


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


def main():
    empty_filename = 'sample_p_dish_images/one pair/picture0.jpg'
    full_filename = 'sample_p_dish_images/one pair/picture1.jpg'

    empty_image = cv.imread(empty_filename)
    full_image = cv.imread(full_filename)
    assert (empty_image is not None) and (full_image is not None)

    empty_cropped = get_cropped_image(empty_image)
    full_cropped = get_cropped_image(full_image)

    # scale images to match
    if empty_cropped.shape[0] != full_cropped.shape[0]: # images are different sizes
        scale_factor = full_cropped.shape[0]/empty_cropped.shape[0]
        empty_cropped = resize_image(empty_cropped, scale_factor)

    cv.imshow("empty_cropped after resize", empty_cropped)
    cv.imshow("full_cropped after resize", full_cropped)
    cv.waitKey(0)

    image_subtraction_approach(empty_cropped, full_cropped)


main()