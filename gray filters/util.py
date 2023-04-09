import cv2
import matplotlib.pyplot as plt 
import math
import numpy as np
import csv

NOISE_PERCENTAGES = [70, 90]
NUMBER_OF_ITERATIONS = 3

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(mse):
    return 10*math.log10(((255)**2)/mse)

def plot_graphs(i, module_name, filtered_image, iterations, mse_values, psnr_values, time_values):
    filtered_image_name = f'{module_name}/filtered_{i}.png'
    cv2.imwrite(filtered_image_name, filtered_image)

    plt.plot(iterations, time_values, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "Average execution time")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Iterations')
    plt.ylabel('Average execution time (second)')
    plt.title('Duration graph', fontsize = 20)
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(f'{module_name}/execution_time_plot_{i}.png')
    plt.close()


    plt.plot(iterations, mse_values, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "MSE values")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Iterations')
    plt.ylabel('MSE values')
    plt.title('MSE calculation', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.savefig(f'{module_name}/mse_values_{i}.png')
    plt.close()


    plt.plot(iterations, psnr_values, color = 'g', linestyle = 'dashed',
            marker = 'o',label = "PSNR values")
    
    plt.xticks(rotation = 10)
    plt.xlabel('Iterations')
    plt.ylabel('PSNR values')
    plt.title('PSNR calculation', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.savefig(f'{module_name}/psnr_values_{i}.png')
    plt.close()


def write_csv(module_name, noise_percentages, mse_values, psnr_values, duration_with_noises):

        csv_file_name = f'{module_name}/comparisons.csv'

        rows = zip(noise_percentages, mse_values, psnr_values, duration_with_noises)

        with open(csv_file_name, 'w') as f:
                write = csv.writer(f)

                print('writing to csv--------------------')
                print(noise_percentages, mse_values, psnr_values)

                for row in rows:
                        write.writerow(row)

