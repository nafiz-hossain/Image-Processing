from skimage.util import random_noise, img_as_int
import cv2
import numpy as np
from math import e
import datetime
import statistics

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
    
    def  __init__(self, noisy_image):
        self.noisy_image = noisy_image
        self.histogram, bin_edges = np.histogram(noisy_image, bins=256, range=(0, 255))
        x, y = noisy_image.shape
        self.cardinality = x*y
        self.threshold_wise_intensity = dict()

        for i in range(0, 255):
            self.__mean_intensity_of_background_and_object(s=i)
        
        
        
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


    def __background_membership_function(self, pixel, s, w):
        if pixel<s:
            mb = self.threshold_wise_intensity[s]['mb']
            return e**(abs(pixel-mb)*(-1)/w)                
        return 0


    def __object_membership_function(self, pixel, s, w):
        if pixel>=s:
            mo = self.threshold_wise_intensity[s]['mo']
            return e**(abs(pixel-mo)*(-1)/w)
        return 0


    def __calculate_entropy(self, s, w):
        summation = 0
        for pixel in range(0, 256):
            hesitation = (1-self.__background_membership_function(pixel=pixel, s=s, w=w)) * (1-self.__object_membership_function(pixel=pixel, s=s, w=w))
            summation = summation + (self.histogram[pixel]*hesitation)
        return (1/self.cardinality)*summation


    def __membership_function_for_noise_detection(self, pixel, a, b):
        if abs(pixel - a) < a:
            return 0
        elif abs(pixel - b) > b:
            return 1
        elif a >= abs(pixel - (a+b)/2) and abs(pixel - (a+b)/2)<b:
            return (pixel-a)/(b-a)


    def __noise_detection_unit(self, t):
        a = self.threshold_wise_intensity[t]['mb']
        b = self.threshold_wise_intensity[t]['mo']

        membership_func = lambda x: self.__membership_function_for_noise_detection(x, a, b)
        vectorized_membership = np.vectorize(membership_func)
        matrix_with_membership = vectorized_membership(self.noisy_image)

        return matrix_with_membership

    
    def noise_detection(self):
        rows, columns = self.noisy_image.shape

        valid_ws = list()
        
        for w in range(224, 225):
            invalid_w = False

            print(w)
            
            for s in range(0, 1):
                membership_func = lambda x: self.__membership_function(x, s, w)
                vectorized_membership = np.vectorize(membership_func)
                matrix_with_membership = vectorized_membership(self.noisy_image)

                if(matrix_with_membership.any()<0.5):
                    invalid_w = True
                    break

                
            if  invalid_w:
                continue
            valid_ws.append(w)
        print(valid_ws)
        entropy_list = list()
        for s in range(0, 255):
            entropies = list()
            for w in valid_ws:
                entropies.append(self.__calculate_entropy(s, w))

            vals, counts = np.unique(entropies, return_counts=True)
            mode_value = np.argwhere(counts == np.max(counts))
            modals = vals[mode_value].flatten().tolist()
            if len(modals)==0:
                least_entropy =  statistics.median(entropies)
            else:
                least_entropy = modals[0]
            entropy_list.append(least_entropy)
        entropy_array = np.array(entropy_list)
        minimum_index = entropy_array.argmin()
        print(minimum_index)

        return self.__noise_detection_unit(t=minimum_index)


    

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
    first_detection = FirstDetectionUnit().first_noise_detection(noisy_image=gray_noisy_image)
    second_detection = SecondDetectionUnit(noisy_image=gray_noisy_image).noise_detection()


    '''noise removal'''
    rows, columns = gray_noisy_image.shape 
    for i in range(1, rows):
        for j in range(1, columns):
            m = 
    
    
    