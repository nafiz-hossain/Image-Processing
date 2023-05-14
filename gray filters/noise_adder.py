from skimage.util import random_noise, img_as_int
import cv2
import numpy as np
import util
NOISE_PERCENTAGES = util.NOISE_PERCENTAGES

if __name__ == "__main__":
    original_image = cv2.imread('image2.jpg', 0)
    for i in NOISE_PERCENTAGES:
        noisy_img = random_noise(original_image, mode='s&p',
                                amount=i/100)
        gray_noisy_image = np.array(255*noisy_img, dtype='uint8')
        noisy_image_name = f'noisy_image_{i}.png'
        cv2.imwrite(noisy_image_name, gray_noisy_image)
