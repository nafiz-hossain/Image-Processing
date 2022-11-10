import cv2
import numpy as np
import statistics
import os
import csv
import time
import pandas as pd  #pip install pandas
import matplotlib.pyplot as plt   #pip install matplotlib


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



def medianFiltering(m,n,old,new,window_size):
    for i in range(1, m+1):
        for j in range(1, n+1):
            try:     
                pos_a= i - int(window_size/2)
                pos_b= j - int(window_size/2)
                pos_c= i + int(window_size/2)
                pos_d= j + int(window_size/2)
                
                temp = old[pos_a:pos_c+1, pos_b:pos_d+1]
                temp = temp.flatten()
                temp = np.delete(temp, int(len(temp)/2))    
                temp = np.median(temp)
                new[i, j]= temp
                #print('temp ', temp)
            except:
                print('temp error for value: ', temp)
                print('Error occurred')
     
    #astype returns a new DataFrame where the data types has been changed to the specified type
    #numpy.uint8: 8-bit unsigned integer (0 to 255).
    new = new.astype(np.uint8)
    final= new[1:m+1 , 1:n+1]
    return final



def main():
    loop = 2
    duration_arr = []
    final_duration = []
    matrix_size_array = []
    #print(os.path.exists('samplev2.png'))    
    #take input by cv2
    img_input = cv2.imread('samplee.png', 1)



    m, n, c = img_input.shape
    matrix_size = 3




    # print('Before sending: m,n',m,n)

    # print('Before sending: b',b)

    # print('Before sending: b_new',b_new)
    for outer in range(0,3,2):
        matrix_size = matrix_size + outer
        pos= int(matrix_size/2)
        updatedImage = np.array([[[0]*3]*(n+(matrix_size-1))]*(m+(matrix_size-1)))
        
        # print('updatedImage shape', updatedImage.shape)
        # print('updatedImage after declaring np array', updatedImage)

        updatedImage[pos:m+pos , pos:n+pos, :] = img_input




        
        b_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
        g_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
        r_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])


        # print('updatedImage after using img_input', updatedImage)
        b, g, r = cv2.split(updatedImage)


        print('Outer is ',outer)
        var = matrix_size + outer
        matrix_size_array.append(var)
        for iter in range(loop):
            start_timer = round(time.time() * 1000)
            print('b is ', b)
            print('b_new is ', b_new)
            
            b_final= medianFiltering(m,n,b,b_new,matrix_size)
            g_final= medianFiltering(m,n,g,g_new,matrix_size)
            r_final= medianFiltering(m,n,r,r_new,matrix_size)
            stop_timer = round(time.time() * 1000)
            
            duration_in_second = stop_timer-start_timer
            duration_arr.append(duration_in_second)
        
        print('matrix_size_array ',matrix_size_array)

        final_img = cv2.merge((b_final,g_final,r_final))
        # print('final_b ', b_final)
        cv2.imwrite('filtered.png', final_img)


        print('##Duration##',duration_arr)
        print('##matrix_size_array##',matrix_size_array)

        #csv part
        column_name =   str(matrix_size) + 'x' + str(matrix_size)
        df_source1 = pd.read_csv('source.csv',header=None) 
        duration = sum(duration_arr)/len(duration_arr)
        final_duration.append(duration)
        df_source1[column_name]=pd.Series(duration)
        df_source1 = df_source1.iloc[: , 1:]
        df_source1 = df_source1.dropna()
        df_source1.to_csv('output.csv',index=False)
        
        print(df_source1)


    #visualize duration based on matrix size
    
    # with open('output.csv','r') as csvfile:
    #     lines = csv.reader(csvfile, delimiter=',')
    #     for row in lines:
    #         x.append(row[0])
    #         y.append(int(row[1]))
    

    print('Final duration', final_duration)
    #print('Matrix size array', matrix_size_array)
    
    plt.plot(matrix_size_array, final_duration, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "Average time")
    
    plt.xticks(rotation = 25)
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (ms)')
    plt.title('Median Filtering', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()



    
    # print('###time', duration_in_second)
    # print('Updated Image to', final_img)
    # print('b_new should be', b_final)
    # print('g_new should be', g_final)
    # print('r_new should be', r_final)


if __name__ == "__main__":
    main()
