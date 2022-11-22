import math
import cv2
import numpy as np
import statistics
import os
import csv
import time
import pandas as pd  #pip install pandas
import matplotlib.pyplot as plt   #pip install matplotlib
from traceback import print_exc

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

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def Calculate_psnr(originalImage, filteredImage):
    # return cv2.PSNR(originalImage,filteredImage)
    return mse(originalImage, filteredImage)



def medianFiltering(m,n,old,new,window_size):
    old = np.array(old[0])
    # print("old image:\n",old)

    window_half = int(window_size/2)
    for i in range(window_half, m+1):
        for j in range(window_half, n+1):
            try:     
                pos_a= i - window_half
                pos_b= j - window_half
                pos_c= i + window_half
                pos_d= j + window_half
                temp = old[pos_a:pos_c+1, pos_b:pos_d+1]
                # print("temp: \n",temp)
                
                temp = temp.flatten()
                #print('tempValue for temp', len(temp), pos_a,pos_c+1, pos_b,pos_d+1, i, j)
                temp = np.delete(temp, int(len(temp)/2))  
                #print('updatedTemp len', len(temp))  
                temp = np.median(temp)
                new[i, j]= temp
                #print('temp ', temp)
            except Exception as e:
                # print('temp error for value: ')
                print_exc(e)
                print('Error occurred ', e)
     
    #astype returns a new DataFrame where the data types has been changed to the specified type
    #numpy.uint8: 8-bit unsigned integer (0 to 255).
    new = new.astype(np.uint8)
    final= new[1:m+1 , 1:n+1]
    return final



def main():

    loop = 1
    duration_arr = []
    average_execution_time = []
    matrix_size_array = []
    psnr_values = []
    matrix_size = 1
    df_source1 = pd.DataFrame([])
    df_source_psnr = pd.DataFrame([])

    number_of_window = 10

    # outer_v2 = []


    #print(os.path.exists('samplev2.png'))    
    #take input by cv2
    img_input = cv2.imread('sample_noise_grayscale_1.png', 0)
    img_original = cv2.imread('sample_original_grayscale_1.png', 0)

    print(img_input.shape)
    m, n = img_input.shape



    for outer in range(0,number_of_window):
        # outer_v2.append(outer)
        matrix_size = matrix_size + 2
        pos = int(matrix_size/2)
        updatedImage = np.array([ [0]*(n+(matrix_size-1))]*(m+(matrix_size-1)))
        
        # print('updatedImage shape', updatedImage.shape)
        
        updatedImage[pos:m+pos , pos:n+pos] = img_input


        # print('updatedImage after declaring np array', updatedImage)

        
        b_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
        

        # print('updatedImage after using img_input', updatedImage)
        b = cv2.split(updatedImage)
        
        filename = str(matrix_size) + 'x' + str(matrix_size)
        print('Outer is ',outer)
        matrix_size_array.append(matrix_size)
        duration_arr.clear()
        for iter in range(loop):
            start_timer = round(time.time() * 1000)
            print(f'Running loop number: {iter}, window size {matrix_size}')
            
            b_final= medianFiltering(m,n,b,b_new,matrix_size)
            
            stop_timer = round(time.time() * 1000)
            
            duration_in_second = stop_timer-start_timer
            duration_arr.append(duration_in_second)
        

        average_execution_time.append(np.average(duration_arr)/1000)

        final_img = b_final
        
        cv2.imwrite('filtered_gray_%s.png'%filename, final_img)
        psnr_values.append(Calculate_psnr(img_original, final_img))

        print('##Duration##',duration_arr)
        #print('##matrix_size_array##',matrix_size_array)

        #csv part
        column_name =   str(matrix_size) + 'x' + str(matrix_size)
        # df_source1 = pd.read_csv('source.csv',header=None) 

        df_source1[column_name]=pd.Series(np.average(duration_arr)/1000)
        df_source_psnr[column_name]=pd.Series(psnr_values)

        # df_source1 = df_source1.iloc[: , 1:]
        
    df_source1 = df_source1.dropna()
    df_source1.to_csv('output_gray.csv',index=False)
    df_source_psnr = df_source1.dropna()
    df_source_psnr.to_csv('psnr_output_gray.csv',index=False)
    print(df_source1)

    print('##matrix_size_array##',matrix_size_array)
    # print('### Outer is ###', outer_v2)
    print('Average execution time ', average_execution_time)
    #visualize duration based on matrix size
    
    # with open('output.csv','r') as csvfile:
    #     lines = csv.reader(csvfile, delimiter=',')
    #     for row in lines:
    #         x.append(row[0])
    #         y.append(int(row[1]))
    

    print('Final duration', average_execution_time)
    #print('Matrix size array', matrix_size_array)

    plt.plot(matrix_size_array, average_execution_time, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "Average execution time")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Window size')
    plt.ylabel('Average execution time (second)')
    plt.title('Median Filtering', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()




    print('PSNR value is ', psnr_values)

    plt.plot(matrix_size_array, psnr_values, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "MSE values")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Window size')
    plt.ylabel('MSE values')
    plt.title('MSE calculation', fontsize = 20)
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
