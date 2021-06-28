import cv2
import numpy as np
from matplotlib import pyplot as plt


def spectral_residual_saliency(image):
    h, w, c = image.shape
    output_saliency = np.zeros((64,64,3))

    for i in range(c):
        img = image[:,:,i]
        img = cv2.resize(img, (64,64),cv2.INTER_LINEAR_EXACT)

        fft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
        amp, phase = cv2.cartToPolar(fft[:,:,0],fft[:,:,1])
        SR = np.exp(np.log(amp) - cv2.boxFilter(np.log(amp), -1, (3,3),cv2.BORDER_DEFAULT))

        fft[:,:,0], fft[:,:,1] = cv2.polarToCart(SR,phase)
        ifft = cv2.dft(fft, flags = cv2.DFT_INVERSE)
        amp, phase = cv2.cartToPolar(ifft[:,:,0], ifft[:,:,1])
        # amp = amp*amp

        amp = cv2.GaussianBlur(amp,(5,5),8,0,cv2.BORDER_DEFAULT)
        amp = amp*amp
        min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(amp) 
        amp = amp/max_val
        output_saliency[:,:,i] = amp

    saliencyMap = 0.333*output_saliency[:,:,0] + 0.333*output_saliency[:,:,1] + 0.333*output_saliency[:,:,2]
    saliencyMap = cv2.resize(saliencyMap, (h,w), cv2.INTER_LINEAR_EXACT)
    saliencyMap = (saliencyMap * 255).astype("uint8") 

    return saliencyMap
