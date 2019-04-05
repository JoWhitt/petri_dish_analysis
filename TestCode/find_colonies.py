import cv2 as cv
import numpy as np

def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return resized


def image_subtraction_approach(empty_dish, full_dish):
    # updated 29/03/2019
    if (empty_dish is None) or (full_dish is None):
        return 0

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
    # updated 29/03/2019
    # the correct way to call this function depends on the version of opencv run
    ret, contours, hierarchy = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
                    cv.imshow('eroded image', roi)
                    contours_2, hierarchy = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours_2:
                    new_contours.append(cnt)
            # cv.drawContours(result_image, [hull], 0, (0,255,0))

			
    mean_val = []
    for cnt in new_contours:
        mask = np.zeros(thresholded.shape,np.uint8)
        cv.drawContours(mask,[cnt],0,255,-1)
        mean_val.append(cv.mean(full_dish, mask=mask))
		
	#red, blue, yellow, gray (in bgr format)	
    red_lower = [0,0,112]
    red_upper = [201,209,255]
    blue_lower = [102,0,0]
    blue_upper = [255,227,175]
    red_count = 0
    blue_count = 0
    other_count = 0
    for i in mean_val:
        if i[0] > red_lower[0] and i[0] <= red_upper[0] and i[1] > red_lower[1] and i[1]<= red_upper[1] and i[2] > red_lower[2] and i[2] <= red_upper[2]:
            red_count+=1
        elif i[0] > blue_lower[0] and i[0] <= blue_upper[0] and i[1] > blue_lower[1] and i[1]<= blue_upper[1] and i[2] > blue_lower[2] and i[2] <= blue_upper[2]:
            blue_count+=1
        else:
            other_count+=1

    colour_count = []
    colour_count.append(red_count)
    colour_count.append(blue_count)
    colour_count.append(other_count)
    print("colours",colour_count)

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

    both_counts = [count, colour_count]

    return both_counts


''' read image (containing one petri dish) display & 
return cropped image with masked out background'''
def get_cropped_image(input_image):
    cropped_image = None

    ''' 1. FIND DISH '''
    # smooth image & convert to grayscale
    smoothed = cv.medianBlur(input_image,5)
    gray = cv.cvtColor(smoothed, cv.COLOR_BGR2GRAY)

    # apply hough algorithm to locate dish 
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100)

    # updated 29/03/2019
    # one and only one dish found
    if (circles is not None)  and (len(circles)==1):
        circles = np.round(circles[0, :]).astype("int")
        (x, y, r) = circles[0]

        ''' 2. CREATE & APPLY MASK '''
        mask = np.zeros(input_image.shape, np.uint8) # initialise image
        cv.circle(mask, (x, y), int(r/2), (255,255,255), r+1) # draw circle into mask
        masked_image = cv.bitwise_and(input_image, mask)

        ''' 3. CROP IMAGE '''
        cropped_image = masked_image[(y-r):(y+r), (x-r):(x+r)]

    return cropped_image



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



def main():
   # empty_filename = '../sample_p_dish_images/one pair/picture0.jpg'
   # full_filename = '../sample_p_dish_images/one pair/picture1.jpg'
    empty_filename = '../sample_p_dish_images/colony_growth_stage_0.jpg'
    full_filename = '../sample_p_dish_images/colony_growth_stage_2.jpg'
    empty_filename = '../sample_p_dish_images/four_empty_dishes.png'
    full_filename = '../sample_p_dish_images/four_full_dishes.png'

    # updated 29/03/2019
    empty_image = cv.imread(empty_filename)
    full_image = cv.imread(full_filename)
    assert (empty_image is not None) and (full_image is not None)

    split_empty_image = split_image(empty_image)
    split_full_image = split_image(full_image)

    zip_image = zip(split_empty_image,split_full_image)
    colony_count = []

    iteration_number = 0
    for i,j in zip_image:
        print ('iteration_number:', iteration_number)

        cv.imshow("quarter image",j)
        cv.waitKey(0)

        assert (i is not None) and (j is not None)
        empty_cropped = get_cropped_image(i)
        full_cropped = get_cropped_image(j)

        # updated 29/03/2019
        # scale images to match
        if (empty_cropped is not None) and (full_cropped is not None):
            if empty_cropped.shape[0] != full_cropped.shape[0]: # images are different sizes
                scale_factor = full_cropped.shape[0]/empty_cropped.shape[0]
                empty_cropped = resize_image(empty_cropped, scale_factor)

        colony_count.append( image_subtraction_approach(empty_cropped, full_cropped) )
       
    print(colony_count)



if __name__ == '__main__':
    main()