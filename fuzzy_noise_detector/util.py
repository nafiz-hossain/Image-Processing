import numpy as np

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# def calculate_psnr(originalImage, filteredImage):
#     # return cv2.PSNR(originalImage,filteredImage)
#     return mse(originalImage, filteredImage)
