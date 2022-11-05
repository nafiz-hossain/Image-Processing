import cv2
import numpy as np
import statistics 


# enum cv::ImreadModes {
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




#The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color): 
#Channels refer to the number of colors. For example, there are three channels in a RGB image, the Red Channel, the Green Channel and the Blue Channel. Each of the channels in each pixel represents the intensity of each color that constitute that pixel.


def medianFiltering(m,n,old,new):
    for i in range(1, m+1):
        for j in range(1, n+1):
            temp = [ old[i-1, j-1],
            old[i-1, j],
            old[i-1, j+1],
            old[i, j-1],
            old[i, j+1],
            old[i+1, j-1],
            old[i+1, j],
            old[i+1, j+1],
            ]
            temp = statistics.median(temp)
            new[i, j]= temp

    #astype returns a new DataFrame where the data types has been changed to the specified type
    #numpy.uint8: 8-bit unsigned integer (0 to 255).
    new = new.astype(np.uint8)
    final= new[1:m+1 , 1:n+1]
    return final


def main():
    #take input by cv2
    img_input = cv2.imread('samplee.png', 1)
    m, n, c = img_input.shape

    # print('input image shape', img_input.shape)
    # print('m= ', m)
    # print('n= ', n)
    # print('c= ', c)

    updatedImage = np.array([[[0]*3]*(n+2)]*(m+2))
    # print('updatedImage shape', updatedImage.shape)
    # print('updatedImage after declaring np array', updatedImage)


    updatedImage[1:m+1 , 1:n+1, :] = img_input

    #print('updatedImage after using img_input', updatedImage)
    b, g, r = cv2.split(updatedImage)




    b_new = np.zeros([m+2, n+2])
    g_new = np.zeros([m+2, n+2])
    r_new = np.zeros([m+2, n+2])


    b_final= medianFiltering(m,n,b,b_new)
    g_final= medianFiltering(m,n,g,g_new)
    r_final= medianFiltering(m,n,r,r_new)

    final_img = cv2.merge((b_final,g_final,r_final))
    cv2.imwrite('filtered.png', final_img)
    # print('Updated Image to', final_img)
    # print('b_new should be', b_final)
    # print('g_new should be', g_final)
    # print('r_new should be', r_final)


if __name__ == "__main__":
    main()


