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
import util

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

    noise_percentages = [10,  50, 90]
    print('noise percentages: ', noise_percentages)
    original_image = cv2.imread('image2.jpg', 0)

    mse_with_noises = []
    psnr_with_noises = []
    duration_with_noises = []

    for p in noise_percentages:
        print(f'now working for image with noise {p}')
        img_input = cv2.imread(f'noisy_image_{p}.png', 0)

        loop = util.NUMBER_OF_ITERATIONS
        duration_arr = []
        average_execution_time = []
        matrix_size_array = []
        psnr_values = []
        mse_values = []
        matrix_size = 1
        df_source1 = pd.DataFrame([])
        df_source_psnr = pd.DataFrame([])
        iterations = []
        number_of_window = 1

        print(img_input.shape)
        m, n = img_input.shape



        for outer in range(0,number_of_window):
            # outer_v2.append(outer)
            matrix_size = matrix_size + 2
            pos = int(matrix_size/2)
            updatedImage = np.array([ [0]*(n+(matrix_size-1))]*(m+(matrix_size-1)))
            
            # print('updatedImage shape', updatedImage.shape)
            
            updatedImage[pos:m+pos , pos:n+pos] = img_input
 
            b_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
           
            b = cv2.split(updatedImage)
            
            filename = str(matrix_size) + 'x' + str(matrix_size)
            print('Outer is ',outer)
            matrix_size_array.append(matrix_size)
            duration_arr.clear()
            for iter in range(loop):
                start_timer = round(time.time() * 1000)
                print(f'Running loop number: {iter}, window size {matrix_size}')
                
                final_img= medianFiltering(m,n,b,b_new,matrix_size)
                
                stop_timer = round(time.time() * 1000)
                
                duration_in_second = (stop_timer-start_timer)/1000
                duration_arr.append(duration_in_second)
            

                mse_value = util.mse(original_image, final_img)
                mse_values.append(mse_value)
                psnr_values.append(util.psnr(mse_value))
                iterations.append(iter)

            
        util.plot_graphs(p, 'median7', final_img, iterations, mse_values, psnr_values, duration_arr)

        mse_avg = np.average(mse_values)
        psnr_avg = np.average(psnr_values)
        avg_duration = np.average(duration_arr)

        mse_with_noises.append(mse_avg)
        psnr_with_noises.append(psnr_avg)
        duration_with_noises.append(avg_duration)
    
    print(noise_percentages)
    print(mse_with_noises)
    print(psnr_with_noises)
    util.write_csv('median7', noise_percentages, mse_with_noises, psnr_with_noises, duration_with_noises)
        
       

        
if __name__ == "__main__":
    main()
