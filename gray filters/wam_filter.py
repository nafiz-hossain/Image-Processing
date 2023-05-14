import cv2
import numpy as np
import math
from skimage.util import random_noise, img_as_int
import util
import time
import matplotlib.pyplot as plt 
import skfuzzy as fuzz
from skimage.morphology import disk, closing, opening

T = 10
WINDOW_SIZE = 7
NOISE_PERCENTAGE = 90
N = 10
beta = 3


def manhattan_distance(i1, j1, i2, j2):
    return abs(i1-i2) + abs(j1-j2)


def euclidean_distance(i1, j1, i2, j2):
    return math.sqrt((i1-i2)**2 + (j1-j2)**2)


def inifinity_distance(i1, j1, i2, j2):
    return max(abs(i1-i2), abs(j1-j2))


def get_uncorrupted_pixels_for_manhattan_distance(d, i, j, noise_detection_frame):

    image_rows, image_columns = noise_detection_frame.shape

    # temp = np.zeros((15, 15))

    uncorrupted_pixel_count = 0

    non_corrupted_pixels = []

    row_start = i-d
    row_end = i-1
    col_start = j
    col_end = j

    # for lines upper i,j pixel
    for x in range(row_start, row_end+1):
        if (x < 0 or x >= image_rows):
            continue
        for y in range(col_start, col_end+1):
            if y < 0 or y >= image_columns:
                continue
            # temp[x, y] = 1
            if (noise_detection_frame[x, y] == 0):
                uncorrupted_pixel_count = uncorrupted_pixel_count+1
                non_corrupted_pixels.append((x, y))

        col_start = col_start-1
        col_end = col_end+1

    # print(temp)
    # print('-----------------------------------------------------------------------')

    col_start = j-d
    col_end = j+d

    # for line of i,j pixel
    for x in range(col_start, col_end+1):
        if x < 0 or x >= image_columns:
            continue

        # temp[i, x] = 1
        if (noise_detection_frame[i, x] == 0):
            uncorrupted_pixel_count = uncorrupted_pixel_count+1
            non_corrupted_pixels.append((i, x))

    # print(temp)
    # print('-----------------------------------------------------------------------')

    row_start = i+1
    row_end = i+d
    col_start = j-d+1
    col_end = j+d-1

    # for lines lower i,j pixel
    for x in range(row_start, row_end+1):
        if (x < 0 or x >= image_rows):
            continue
        for y in range(col_start, col_end+1):
            if y < 0 or y >= image_columns:
                continue

            # temp[x, y] = 1
            if (noise_detection_frame[x, y] == 0):
                uncorrupted_pixel_count = uncorrupted_pixel_count+1
                non_corrupted_pixels.append((x, y))

        col_start = col_start+1
        col_end = col_end-1

    # print(temp)

    return non_corrupted_pixels


def get_alpha_value(base_pixel, neighbour_pixel):
    alpha = 1 / (beta**manhattan_distance(base_pixel[0], base_pixel[1],
         neighbour_pixel[0], neighbour_pixel[1]))
    return alpha


def weighted_arithmetic_mean(center_pixel, non_corrupted_pixels, image):

    all_pixels_alpha_value = [get_alpha_value(
        center_pixel, pixel) for pixel in non_corrupted_pixels]
    alpha_sum = sum(all_pixels_alpha_value)

    summation = 0

    for pixel in non_corrupted_pixels:
        weight = get_alpha_value(center_pixel, pixel)/alpha_sum
        summation = summation + (image[pixel[0], pixel[1]] * weight)

    return summation


def filter_image(noisy_image, noisy_pixels, noise_detection_frame):

    # print(noisy_pixels)
    # print(noise_detection_frame)
    
    # print(noisy_image)
    
    for pixel in noisy_pixels:
        row = pixel[0]
        column = pixel[1]

        D = 1
        non_corrupted_pixels = []
        while (1):
            non_corrupted_pixels = get_uncorrupted_pixels_for_manhattan_distance(
                d=D, i=row, j=column, noise_detection_frame=noise_detection_frame)

            if len(non_corrupted_pixels) >= N:
                break

            D = D+1

        # print(f'age chilo value {noisy_image[row, column]}')
        noisy_image[row, column] = weighted_arithmetic_mean(
            center_pixel=pixel, non_corrupted_pixels=non_corrupted_pixels, image=noisy_image)
        # print(f'pore hoise value {noisy_image[row, column]}')
    
    # print(noisy_image)
    # cv2.imwrite('filtered_image_new.png', noisy_image)
        
    return noisy_image


def get_window_max_min_val(image, i, j):
    half_window = int(WINDOW_SIZE/2)

    start_row = i - half_window
    end_row = i + half_window
    start_col = j - half_window
    end_col = j + half_window

    (rows, columns) = image.shape

    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0
    if end_row > rows:
        end_row = rows
    if end_col > columns:
        end_col = columns

    window = image[start_row: end_row+1, start_col: end_col+1]

    return np.max(window), np.min(window)


def is_noisy(pixel_value, opening_closing, closing_opening, s_max, s_min):
    d = (opening_closing + closing_opening)/2 - pixel_value

    if d >= T and (pixel_value == s_max or pixel_value == s_min):
        b = 255
    else:
        b = 0

    if b == 255:
        # print('found noisy!!!')
        return 1
    else:
        # print('Not found noisy!!!')
        return 0


def detect_and_filter_noise(image):
    
    new_image = image.copy()

    structuring_element_size = (5, 5)
    # kernel = cv2.getStructuringElement(
    #     cv2.MORPH_RECT, structuring_element_size)

    # opening = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel)
    # opening_closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # closing = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
    # closing_opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    kernel = disk(5)
    
    closed = closing(new_image, kernel)
    closing_opening = opening(closed, kernel)

    opened = opening(new_image, kernel)
    opening_closing = closing(opened, kernel)


    (rows, columns) = new_image.shape

    noise_detection_frame = np.zeros((rows, columns), dtype=int)

    noisy_pixels = []

    for i in range(0, rows):
        for j in range(0, columns):
            s_max, s_min = get_window_max_min_val(new_image, i, j)
            # print(s_max, s_min)
            noise_detection_frame[i, j] = is_noisy(
                new_image[i, j], opening_closing[i, j], closing_opening[i, j], s_max, s_min)
            if noise_detection_frame[i, j] == 1:
                noisy_pixels.append((i, j))

    # print(noise_detection_frame)
    # cv2.imshow("noisy image", gray)

    return filter_image(noisy_image=new_image, noisy_pixels=noisy_pixels,
                 noise_detection_frame=noise_detection_frame)

    # cv2.imshow("noisy image", gray)


if __name__ == "__main__":

    noise_percentages = util.NOISE_PERCENTAGES
    # noise_percentages = [50]
    print('noise percentages: ', noise_percentages)
    original_image = cv2.imread('image2.jpg', 0)

    mse_with_noises = []
    psnr_with_noises = []
    duration_with_noises = []
    
    for p in noise_percentages:
        print(f'now working for image with noise {p}')
        gray_noisy_image = cv2.imread(f'noisy_image_{p}.png', 0)
    
        duration_arr = []
        iterations = []
        average_execution_time = []
        psnr_values = []
        mse_values = [] 

        for i in range(0,util.NUMBER_OF_ITERATIONS):

            start_timer = round(time.time() * 1000)
            filtered_image = detect_and_filter_noise(image=gray_noisy_image)
            
            stop_timer = round(time.time() * 1000)        
            duration_in_second = (stop_timer-start_timer)/1000
            duration_arr.append(duration_in_second)
            iterations.append(i)

            mse_value = util.mse(original_image, filtered_image)
            mse_values.append(mse_value)
            psnr_values.append(util.psnr(mse_value))

        util.plot_graphs(p, 'wam', filtered_image, iterations, mse_values, psnr_values, duration_arr)
        cv2.imshow('Example_Image.png', filtered_image)
        cv2.waitKey(0)

        mse_avg = np.average(mse_values)
        psnr_avg = np.average(psnr_values)
        avg_duration = np.average(duration_arr)

        mse_with_noises.append(mse_avg)
        psnr_with_noises.append(psnr_avg)
        duration_with_noises.append(avg_duration)
    
    print(noise_percentages)
    print(mse_with_noises)
    print(psnr_with_noises)
    util.write_csv('wam', noise_percentages, mse_with_noises, psnr_with_noises, duration_with_noises)
        