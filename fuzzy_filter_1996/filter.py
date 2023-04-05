import numpy as np
import cv2
from skimage.util import random_noise, img_as_int
import util
import time
import matplotlib.pyplot as plt 

NOISE_PERCENTAGE = 10
NUMBER_OF_ITERATIONS = 10

L = 256
Cpo = (L-1)
Cne = -L+1
Wpo = 2*(L-1)
Wne = 2*(L-1)

a=40
b=32

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

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
    original_image = cv2.imread('5.3.01.tiff', 0)
    noisy_img = random_noise(original_image, mode='s&p',
                             amount=NOISE_PERCENTAGE/100)
    gray_noisy_image = np.array(255*noisy_img, dtype='uint8')
    cv2.imwrite('noisy_image.png', gray_noisy_image)

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

    for k in range(0, NUMBER_OF_ITERATIONS):
        start_timer = round(time.time() * 1000)

        for i in range(1, rows-1):
            for j in range(1, columns-1):
                temp_po = matrix_with_po_membership[i-1:i+2, j-1:j+2]
                temp_ne = matrix_with_ne_membership[i-1:i+2, j-1:j+2]
                lambda1 = rule_base_value(temp_po) 
                lambda2 = rule_base_value(temp_ne)
                lambda0 = max([0, 1-lambda1-lambda2])
                y = (L-1)*(lambda1-lambda2)/(lambda0+lambda1+lambda2)
                y_prime = y*(1-membership_function2(abs(y)))

                filtered_image[i][j] = gray_noisy_image[i][j]+y_prime

        stop_timer = round(time.time() * 1000)        
        duration_in_second = stop_timer-start_timer
        duration_arr.append(duration_in_second/1000)
        iterations.append(k)
          
        cv2.imwrite('filtered.png', filtered_image)
        mse_value = mse(gray_noisy_image, filtered_image)
        mse_values.append(mse_value)

    plt.plot(iterations, duration_arr, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "Average execution time")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Iterations')
    plt.ylabel('Average execution time (second)')
    plt.title('Duration graph 1996', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('execution_time_plot.png')


    plt.plot(iterations, mse_values, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "MSE values")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Iterations')
    plt.ylabel('MSE values')
    plt.title('MSE calculation 1996', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('mse_values.png')

    print(mse_value)

