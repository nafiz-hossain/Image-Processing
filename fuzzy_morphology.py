import cv2 as CV  
import numpy as nmp   

kernel = nmp.ones((5, 5), nmp.uint8)  
img_input = CV.imread('morpho_pic.png', 1)

hsv = CV.cvtColor(img_input, CV.COLOR_BGR2HSV)  
       
# here, we will define the range of masking  
blue_1 = nmp.array([111, 52, 52])  
blue_2 = nmp.array([131, 257, 257])  
       
# Now, we will initialize the mask forconvoluting over input image  
mask = CV.inRange(hsv, blue_1, blue_2)  
res = CV.bitwise_and(img_input, img_input, mask = mask)  

opening = CV.morphologyEx(mask, CV.MORPH_OPEN, kernel)  

while(1):
	#CV.imshow('Mask', mask)  
	CV.imshow('Opening', opening)  

cv2.imwrite('morphology_output.png', opening)
       
