from skimage.util import random_noise, img_as_int
import cv2
import numpy as np
from math import e
import datetime

NOISE_PERCENTAGE = 10

class FirstDetectionUnit:
    KERNEL_1 = np.array([np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]),])
    KERNEL_2 = np.array([np.array([0, -1, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]),])
    KERNEL_3 = np.array([np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([0, 0, 0]),])
    KERNEL_4 = np.array([np.array([0, 0, 0]), np.array([0, 1, -1]), np.array([0, 0, 0]),])
    KERNEL_5 = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1]),])
    KERNEL_6 = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),])
    KERNEL_7 = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]),])
    KERNEL_8 = np.array([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 0, 0]),])


    def first_noise_detection(self, noisy_image):

        '''Normalising'''
        gray_noisy_image = noisy_image/ 255.0

        
        '''convolution operation'''
        convo_1 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_1))
        convo_2 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_2))
        convo_3 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_3))
        convo_4 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_4))
        convo_5 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_5))
        convo_6 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_6))
        convo_7 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_7))
        convo_8 = np.absolute(cv2.filter2D(src=gray_noisy_image, ddepth=-1, kernel=self.KERNEL_8))
        

        N_I = np.minimum(convo_1, convo_2)
        N_I = np.minimum(convo_3, N_I)
        N_I = np.minimum(convo_4, N_I)
        N_I = np.minimum(convo_5, N_I)
        N_I = np.minimum(convo_6, N_I)
        N_I = np.minimum(convo_7, N_I)
        N_I = np.minimum(convo_8, N_I)
        
        
        '''membership value'''
        t = 100
        N_I[N_I > (t/255)] = 1
        N_I[N_I < (t/255)] = 0
        print(N_I)

        return N_I

class SecondDetectionUnit:
    histogram = np.zeros(256)
    threshold_wise_intensity = dict()


    def __calculate_pixel_wise_count(self, image):
        self.histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
        
    def __mean_intensity_of_background_and_object(self, s):
        lob = 0
        hor = 0
        for i in range(0, s+1):
            lob = lob + (i*self.histogram[i])
            hor = hor + self.histogram[i]
        mb = lob/hor

        lob = 0
        hor = 0
        for i in range(s+1, 256):
            lob = lob + (i*self.histogram[i])
            hor = hor + self.histogram[i]

        mo = lob/hor
        self.threshold_wise_intensity[s] = {
            'mb': mb,
            'mo': mo
        } 

    
    
    
    def __membership_function(self, pixel, s, w):
        if pixel<s:
            mb = self.threshold_wise_intensity[s]['mb']
            return e**(abs(pixel-mb)*(-1)/w)
        else:        
            mo = self.threshold_wise_intensity[s]['mo']
            return e**(abs(pixel-mo)*(-1)/w)


    def noise_detection(self, noisy_image):
        self.__calculate_pixel_wise_count(image=noisy_image)

        for i in range(0, 255):
            self.__mean_intensity_of_background_and_object(s=i)
        
        rows, columns = noisy_img.shape

        valid_ws = list()
        
        for w in range(224, 225):
            invalid_w = False
            
            for s in range(0, 255):
                membership_func = lambda x: self.__membership_function(x, s, w)
                vectorized_membership = np.vectorize(membership_func)
                matrix_with_membership = vectorized_membership(noisy_image)

                if(matrix_with_membership.any()<0.5):
                    invalid_w = True
                    break

                # membership = np.empty((rows, columns))
                # invalid_w = False
                # for i in range(0, rows):
                #     for j in range(0, columns):
                #         membership[i][j] = self.__membership_function(pixel=noisy_image[i][j], s=s, w=w)
                #         if membership[i][j]< 0.5:
                #             invalid_w = True
                #             break
                #     if invalid_w:
                #         break
                # if invalid_w:
                #     break
            if  invalid_w:
                continue
            valid_ws.append(w)
        print(valid_ws)

if __name__ == "__main__":
    # start_time = datetime.now()
    original_image = cv2.imread('5.3.01.tiff', 0)
    noisy_img = random_noise(original_image, mode='s&p', amount=NOISE_PERCENTAGE/100)
    gray_noisy_image = np.array(255*noisy_img, dtype = 'uint8')
    print(gray_noisy_image)

    
    
    '''padding 0 srounding image'''
    # padding = 1
    # x_shape, y_shape = gray_noisy_image.shape
    # padded_noisy_image = np.zeros((x_shape+padding*2, y_shape+padding*2))
    # padded_noisy_image[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = gray_noisy_image
    # print(gray_noisy_image)
    # print(x_shape,y_shape)

    second_detection_model = SecondDetectionUnit()
    second_detection_model.noise_detection(noisy_image=gray_noisy_image)

    
    