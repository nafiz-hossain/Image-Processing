import numpy as np
import cv2
import math

def get_median_neighbor_matrix(temp):
    W0 = temp[2,2]
    W1 = temp[0,0]
    W2 = temp[0,1]
    W3 = temp[0,2]
    W4 = temp[0,3]
    W5 = temp[0,4]
    W6 = temp[1,0]
    W7 = temp[1,1]
    W8 = temp[1,2]
    W9 = temp[1,3]
    W10 = temp[1,4]
    W11 = temp[2,0]
    W12 = temp[2,1]
    W13 = temp[2,3]
    W14 = temp[2,4]
    W15 = temp[3,0]
    W16 = temp[3,1]
    W17 = temp[3,2]
    W18 = temp[3,3]
    W19 = temp[3,4]
    W20 = temp[4,0]
    W21 = temp[4,1]
    W22 = temp[4,2]
    W23 = temp[4,3]
    W24 = temp[4,4]

    M1 = np.median([W0, W1, W2, W3, W6, W7, W8, W11, W12])
    M2 = np.median([W0, W2, W3, W4, W7, W8, W9, W12, W13])
    M3 = np.median([W0, W3, W4, W5, W8, W9, W10, W13, W14])
    M4 = np.median([W0, W6, W7, W8, W11, W12, W15, W16, W17])
    M5 = np.median([W0, W8, W9, W10, W13, W14, W17, W18, W19])
    M6 = np.median([W0, W11, W12, W15, W16, W17, W20, W21, W22])
    M7 = np.median([W0, W12, W13, W16, W17, W18, W21, W22, W23])
    M8 = np.median([W0, W13, W14, W17, W18, W19, W22, W23, W24])

    median_matrix = [[M1, M2, M3],[M4, W0, M5], [M6, M7, M8]]

    return np.array(median_matrix)


def get_D_array(median_matrix):
    D = []
    for i in range(3):
        for j in range(3):
            if(i==1 and j==1):
                continue
            D.append(abs(median_matrix[i,j]-median_matrix[1,1]))  
    return D

def fuzzy_inference(D,):
    noisiness = None
    VL1, VL2, VL3, VL4, VL4, VL6, VL7, VL8, VL9, VL10, VL11, VL12, VL13, VL14, VL15, VL16 = 0
    VH1, VH2, VH3, VH4, VH4, VH6, VH7, VH8, VH9, VH10, VH11, VH12, VH13, VH14, VH15, VH16 =1

    if(D[1]<127.5 and D[2]< 127.5  and D[3]<127.5 and D[4]<127.5):
        noisiness = VL
    elif(D[1]<127.5 and D[2]< 127.5  and D[3]<127.5 and D[4]>127.5):
        noisiness = VL
    elif(D[1]<127.5 and D[2]< 127.5  and D[3]>127.5 and D[4]<127.5):
        noisiness = VL
    elif(D[1]<127.5 and D[2]< 127.5  and D[3]>127.5 and D[4]>127.5):
        noisiness = VH

def noise_detect(m,n,old,new,window_size):
    for i in range(2, m+1):
        for j in range(2, n+1):
            pos_a= i - int(window_size/2)
            pos_b= j - int(window_size/2)
            pos_c= i + int(window_size/2)
            pos_d= j + int(window_size/2)
            
            temp = old[pos_a:pos_c+1, pos_b:pos_d+1]

            print("window: ", temp)

            median_matrix = get_median_neighbor_matrix(temp)

            print("median-matrix: \n", median_matrix)

            Darr = get_D_array( median_matrix)

            print("D array: ", Darr)

def get_difference_low_mf(x):
    c = 0
    s = 250
    sub_res = 0.5*(((x-c)/s)^2)
    return math.exp(sub_res)

def get_difference_high_mf(x):
    c = 250
    s = 250
    sub_res = 0.5*(((x-c)/s)^2)
    return math.exp(sub_res)


if __name__ == "__main__":
    
    img_input = cv2.imread('samplee.png', 1)
    m, n, c = img_input.shape
    matrix_size = 5
    pos= int(matrix_size/2)
    print('input image shape', img_input.shape)
    print('Input Image', img_input)

    print('m= ', m)
    print('n= ', n)
    print('c= ', c)

    updatedImage = np.array([[[0]*3]*(n+(matrix_size-1))]*(m+(matrix_size-1)))
    updatedImage[pos:m+pos , pos:n+pos, :] = img_input

    b_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
    g_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])
    r_new = np.zeros([m+(matrix_size-1), n+(matrix_size-1)])


    # print('updatedImage after using img_input', updatedImage)
    b, g, r = cv2.split(updatedImage)
    print('updatedImage after using img_input\n', b)
    

    noise_detect(m,n,b,b_new,matrix_size)


