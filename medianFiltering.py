import cv2
import numpy as np
import statistics 


# enum  	cv::ImreadModes {
#   cv::IMREAD_UNCHANGED = -1,
#   cv::IMREAD_GRAYSCALE = 0,
#   cv::IMREAD_COLOR = 1,
#   cv::IMREAD_ANYDEPTH = 2,
#   cv::IMREAD_ANYCOLOR = 4,
#   cv::IMREAD_LOAD_GDAL = 8,
#   cv::IMREAD_REDUCED_GRAYSCALE_2 = 16,
#   cv::IMREAD_REDUCED_COLOR_2 = 17,
#   cv::IMREAD_REDUCED_GRAYSCALE_4 = 32,
#   cv::IMREAD_REDUCED_COLOR_4 = 33,
#   cv::IMREAD_REDUCED_GRAYSCALE_8 = 64,
#   cv::IMREAD_REDUCED_COLOR_8 = 65,
#   cv::IMREAD_IGNORE_ORIENTATION = 128
# }




#take input by cv2
img_input = cv2.imread('sample.png', 0)
m, n = img_input.shape

print(img_input.shape)


#take a same size matrix set all value into zero
img_new = np.zeros([m, n])
#print(img_new)


print(img_input[0,0])
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [ img_input[i-1, j-1],
        img_input[i, j-1],
        img_input[i+1, j-1],
        img_input[i-1, j],
        img_input[i, j],
        img_input[i+1, j],
        img_input[i-1, j+1],
        img_input[i, j+1],
        img_input[i+1, j+1],
        ]

        temp = statistics.median(temp)
        img_new[i, j]= temp
 
#astype returns a new DataFrame where the data types has been changed to the specified type
#numpy.uint8: 8-bit unsigned integer (0 to 255).

img_new1 = img_new.astype(np.uint8)
cv2.imwrite('filtered.png', img_new1)

