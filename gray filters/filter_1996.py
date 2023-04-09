import numpy as np
import cv2
from skimage.util import random_noise, img_as_int
import util
import time
import matplotlib.pyplot as plt 

# NOISE_PERCENTAGE = 10
# NUMBER_OF_ITERATIONS = 10

L = 256
Cpo = (L-1)
Cne = -L+1
Wpo = 2*(L-1)
Wne = 2*(L-1)

a=40
b=32


def rule_base_value(temp):
    x1 = temp[0][0]
    x2 = temp[0][1]
    x3 = temp[0][2]
    x4 = temp[1][0]
    x5 = temp[1][2]
    x6 = temp[2][0]
    x7 = temp[2][1]
    x8 = temp[2][2]

    value = max([min([x2, x5, x7]), min([x5, x7, x4]), 
        min([x7, x4, x2]), min([x4, x2, x5]), min([x1, x3, x8, x6]),
        min([x1, x2, x3,x5]), min([x2, x3, x5, x8]), min([x3, x5, x8, x7]),
        min([x5, x8, x7, x6]), min([x8, x7, x6, x4]), min([x7, x6, x4, x1]),
        min([x6, x4, x1, x2]), min([x4, x1, x2, x3])])

    return value


def membership_function(u, c, w):
    if u <= (c-w):
        return 0
    elif (c-w) < u and u < (c+w):
        return (w - abs(u-c))/w
    else:
        return 0

def membership_function2(u):
    if u<=a:
        return 1
    elif a<u and u<=(a+b):
        return (a+b-u)/b
    else:
        return 0


if __name__ == "__main__":

    noise_percentages = util.NOISE_PERCENTAGES
    print('noise percentages: ', noise_percentages)
    original_image = cv2.imread('5.3.01.tiff', 0)

    mse_with_noises = []
    psnr_with_noises = []
    duration_with_noises = []

    
    for p in noise_percentages:
        print(f'now working for image with noise {p}')
        gray_noisy_image = cv2.imread(f'noisy_image_{p}.png', 0)
        
        def membership_func(x): return membership_function(x, Cpo, Wpo)
        vectorized_membership = np.vectorize(membership_func)
        matrix_with_po_membership = vectorized_membership(gray_noisy_image)

        def membership_func(x): return membership_function(x, Cne, Wne)
        vectorized_membership = np.vectorize(membership_func)
        matrix_with_ne_membership = vectorized_membership(gray_noisy_image)

        rows, columns = gray_noisy_image.shape
        filtered_image = np.zeros((rows, columns))
        duration_arr = []
        mse_values = []
        iterations = []
        psnr_values = []

        for k in range(0, util.NUMBER_OF_ITERATIONS):
            start_timer = round(time.time() * 1000)

            for i in range(1, rows-1):
                for j in range(1, columns-1):
                    temp_po = matrix_with_po_membership[i-1:i+2, j-1:j+2]
                    temp_ne = matrix_with_ne_membership[i-1:i+2, j-1:j+2]
                    lambda1 = rule_base_value(temp_po) 
                    lambda2 = rule_base_value(temp_ne)
                    lambda0 = max([0, 1-lambda1-lambda2])
                    y = (L-1)*((lambda1-lambda2)/(lambda0+lambda1+lambda2))
                    y_prime = y*(1-membership_function2(abs(y)))

                    filtered_image[i][j] = gray_noisy_image[i][j]+y_prime

            stop_timer = round(time.time() * 1000)        
            duration_in_second = stop_timer-start_timer
            duration_arr.append(duration_in_second/1000)
            iterations.append(k)
            
            mse_value = util.mse(original_image, filtered_image)
            mse_values.append(mse_value)
            psnr_values.append(util.psnr(mse_value))

        util.plot_graphs(p, '1996', filtered_image, iterations, mse_values, psnr_values, duration_arr)

        mse_avg = np.average(mse_values)
        psnr_avg = np.average(psnr_values)
        avg_duration = np.average(duration_arr)

        mse_with_noises.append(mse_avg)
        psnr_with_noises.append(psnr_avg)
        duration_with_noises.append(avg_duration)
    
    print(noise_percentages)
    print(mse_with_noises)
    print(psnr_with_noises)
    util.write_csv('1996', noise_percentages, mse_with_noises, psnr_with_noises, duration_with_noises)
        