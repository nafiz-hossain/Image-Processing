import numpy as np
import cv2
import math

def get_trapezoidal_mf(a, b, c, d):
    temp1 = (x-a)/(b-a)
    temp2 = (d-x)/(d-c)
    temp3 = min([temp1,1, temp2])
    result = max(temp3, 0)
    return result


def get_DL_mf(x):
    c = 0
    s = 250
    sub_res = 0.5*(((x-c)/s)^2)
    return math.exp(sub_res)

def get_DH_mf(x):
    c = 250
    s = 250
    sub_res = 0.5*(((x-c)/s)^2)
    return math.exp(sub_res)


def get_VL_mf(x):
    a = 0
    b = 0
    c = 0.25
    d = 0.75
    
    return reget_trapezoidal_mf(a, b, c, d)


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

def fuzzy_inference(d1_mf_val_DH, d1_mf_val_DL, d2_mf_val_DH, d2_mf_val_DL, d3_mf_val_DH, d3_mf_val_DL, d4_mf_val_DH, d4_mf_val_DL):
    noisiness = None
    
    VL_arr = []
    VH_arr = []
    

    
    r1_strenght = min([d1_mf_val_DL, d2_mf_val_DL, d3_mf_val_DL, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r1_strenght))
    VH_arr.append(0)

    r2_strenght = min([d1_mf_val_DL, d2_mf_val_DL, d3_mf_val_DL, d4_mf_val_DH])
    VL_arr.append(get_VL_mf(r2_strenght))
    VH_arr.append(0)

    r3_strenght = min([d1_mf_val_DL, d2_mf_val_DL, d3_mf_val_DH, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r3_strenght))
    VH_arr.append(0)

    r4_strenght = min([d1_mf_val_DL, d2_mf_val_DL, d3_mf_val_DH, d4_mf_val_DH])
    VL_arr.append(get_VL_mf(r4_strenght))
    VH_arr.append(get_VH_mf(r4_strenght))

    r5_strenght = min([d1_mf_val_DL, d2_mf_val_DH, d3_mf_val_DL, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r5_strenght))
    VH_arr.append(0)

    r6_strenght = min([d1_mf_val_DL, d2_mf_val_DH, d3_mf_val_DL, d4_mf_val_DH])
    VL_arr.append(get_VL_mf(r6_strenght))
    VH_arr.append(get_VH_mf(r6_strenght))

    r7_strenght = min([d1_mf_val_DL, d2_mf_val_DH, d3_mf_val_DH, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r7_strenght))
    VH_arr.append(get_VH_mf(r7_strenght))

    r8_strenght = min([d1_mf_val_DL, d2_mf_val_DH, d3_mf_val_DH, d4_mf_val_DH])
    VL_arr.append(0)
    VH_arr.append(get_VH_mf(r8_strenght))

    r9_strenght = min([d1_mf_val_DH, d2_mf_val_DL, d3_mf_val_DL, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r9_strenght))
    VH_arr.append(0)

    r10_strenght = min([d1_mf_val_DH, d2_mf_val_DL, d3_mf_val_DL, d4_mf_val_DH])
    VL_arr.append(get_VL_mf(r10_strenght))
    VH_arr.append(get_VH_mf(r10_strenght))

    r11_strenght = min([d1_mf_val_DH, d2_mf_val_DL, d3_mf_val_DH, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r11_strenght))
    VH_arr.append(get_VH_mf(r11_strenght))

    r12_strenght = min([d1_mf_val_DH, d2_mf_val_DL, d3_mf_val_DH, d4_mf_val_DH])
    VL_arr.append(0)
    VH_arr.append(get_VH_mf(r12_strenght))

    r13_strenght = min([d1_mf_val_DH, d2_mf_val_DH, d3_mf_val_DL, d4_mf_val_DL])
    VL_arr.append(get_VL_mf(r13_strenght))
    VH_arr.append(get_VH_mf(r13_strenght))

    r14_strenght = min([d1_mf_val_DH, d2_mf_val_DH, d3_mf_val_DL, d4_mf_val_DH])
    VL_arr.append(0)
    VH_arr.append(get_VH_mf(r14_strenght))

    r15_strenght = min([d1_mf_val_DH, d2_mf_val_DH, d3_mf_val_DH, d4_mf_val_DL])
    VL_arr.append(0)
    VH_arr.append(get_VH_mf(r15_strenght))

    r16_strenght = min([d1_mf_val_DH, d2_mf_val_DH, d3_mf_val_DH, d4_mf_val_DH])
    VL_arr.append(0)
    VH_arr.append(get_VH_mf(r16_strenght))

    return VL_arr,VH_arr

def defuzzify(VL_arr, VH_arr, D_arr):
    sum=0
    for l in range(0,16):
        ceter_of_output = (VL_arr[l]+VH_arr)/2
        prod = 1
        for i in range(0,8):




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


            d1_mf_val_DH, d1_mf_val_DL, d2_mf_val_DH, d2_mf_val_DL, d3_mf_val_DH, d3_mf_val_DL, d4_mf_val_DH, d4_mf_val_DL = get_all_mf_values(D)

            print("D array: ", Darr)

def get_all_mf_values(D):
    d1_mf_val_DH = get_DH_mf(D[0])
    d1_mf_val_DL = get_DL_mf(D[0])

    d2_mf_val_DH = get_DH_mf(D[1])
    d2_mf_val_DL = get_DL_mf(D[1])

    d3_mf_val_DH = get_DH_mf(D2])
    d3_mf_val_DL = get_DL_mf(D[2])

    d4_mf_val_DH = get_DH_mf(D[3])
    d4_mf_val_DL = get_DL_mf(D[3])

    return d1_mf_val_DH, d1_mf_val_DL, d2_mf_val_DH, d2_mf_val_DL, d3_mf_val_DH, d3_mf_val_DL, d4_mf_val_DH, d4_mf_val_DL

def get_VH_mf(x):
    a = 0.25
    b = 0.75
    c = 1
    d = 1
    
    return reget_trapezoidal_mf(a, b, c, d)


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


