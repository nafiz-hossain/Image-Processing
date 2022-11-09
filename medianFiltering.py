import cv2
import numpy as np
import statistics
import os

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


def medianFilteringThreeByThree(m,n,old,new):
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

def medianFilteringFiveByFive(m,n,old,new):
    for i in range(2, m+2):
        for j in range(2, n+2):
            temp = [ 
            old[i-2, j-2],
            old[i-2, j-1],
            old[i-2, j],
            old[i-2, j+1],
            old[i-2, j+2],

            old[i-1, j-2],
            old[i-1, j-1],
            old[i-1, j],
            old[i-1, j+1],
            old[i-1, j+2],

            old[i, j-2],
            old[i, j-1],
            old[i, j+1],
            old[i, j+2],

            old[i+1, j-2],
            old[i+1, j-1],
            old[i+1, j],
            old[i+1, j+1],
            old[i+1, j+2],

            old[i+2, j-2],
            old[i+2, j-1],
            old[i+2, j],
            old[i+2, j+1],
            old[i+2, j+2],
            ]
            #print('Every temp: ', temp)
            temp = statistics.median(temp)
            #print('statistics.median(temp): ', temp)
            new[i, j]= temp

    #astype returns a new DataFrame where the data types has been changed to the specified type
    #numpy.uint8: 8-bit unsigned integer (0 to 255).
    new = new.astype(np.uint8)
    final= new[2:m+2 , 2:n+2]
    return new


def medianFilteringSevenBySeven(m,n,old,new):
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
    print(os.path.exists('samplev2.png'))    
    #take input by cv2
    img_input = cv2.imread('samplee.png', 1)
    m, n, c = img_input.shape
    matrix_size = 7
    pos= int(matrix_size/2)
    print('input image shape', img_input.shape)
    print('Input Image', img_input)

    print('m= ', m)
    print('n= ', n)
    print('c= ', c)

    updatedImage = np.array([[[0]*3]*(n+(matrix_size-1))]*(m+(matrix_size-1)))
    
    print('updatedImage shape', updatedImage.shape)
    print('updatedImage after declaring np array', updatedImage)

    updatedImage[pos:m+pos , pos:n+pos, :] = img_input
    
    b_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
    g_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
    r_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])


    print('updatedImage after using img_input', updatedImage)
    b, g, r = cv2.split(updatedImage)




    print('Before sending: m,n',m,n)

    print('Before sending: b',b)

    print('Before sending: b_new',b_new)

    if (matrix_size==3):
        b_final= medianFilteringThreeByThree(m,n,b,b_new)
        g_final= medianFilteringThreeByThree(m,n,g,g_new)
        r_final= medianFilteringThreeByThree(m,n,r,r_new)

    if (matrix_size==5):
        b_final= medianFilteringSevenBySeven(m,n,b,b_new)
        g_final= medianFilteringSevenBySeven(m,n,g,g_new)
        r_final= medianFilteringSevenBySeven(m,n,r,r_new)

    if (matrix_size==7):
        b_final= medianFilteringFiveByFive(m,n,b,b_new)
        g_final= medianFilteringFiveByFive(m,n,g,g_new)
        r_final= medianFilteringFiveByFive(m,n,r,r_new)

    
    final_img = cv2.merge((b_final,g_final,r_final))
    #print('final_b ', b_final)
    cv2.imwrite('filtered.png', final_img)
    # print('Updated Image to', final_img)
    # print('b_new should be', b_final)
    # print('g_new should be', g_final)
    # print('r_new should be', r_final)


if __name__ == "__main__":
    main()


