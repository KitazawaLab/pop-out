import numpy as np
import cv2
from matplotlib import pyplot as plt

def SpectraResidualSaliency(image):
    h, w, c = image.shape
    output_saliency = np.zeros((64,64,3))

    for i in range(3):
        img = image[:,:,i]
        img = cv2.resize(img, (64,64),cv2.INTER_LINEAR_EXACT)

        fft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
        amp, phase = cv2.cartToPolar(fft[:,:,0],fft[:,:,1])
        amp = np.exp(np.log(amp) - cv2.boxFilter(np.log(amp), -1, (3,3),cv2.BORDER_DEFAULT))

        fft[:,:,0], fft[:,:,1] = cv2.polarToCart(amp,phase)
        ifft = cv2.dft(fft, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
        amp, phase = cv2.cartToPolar(ifft[:,:,0], ifft[:,:,1])
        amp = amp**2

        amp = cv2.GaussianBlur(amp,(3,3),8,0,cv2.BORDER_DEFAULT)
        amp = cv2.normalize(amp, 0., 1., cv2.NORM_MINMAX)
        output_saliency[:,:,i] = amp

    saliencyMap = 0.333*output_saliency[:,:,0] + 0.333*output_saliency[:,:,1] + 0.333*output_saliency[:,:,2]
    saliencyMap = cv2.resize(saliencyMap, (h,w), cv2.INTER_LINEAR_EXACT)
    saliencyMap = cv2.resize(saliencyMap, (h,w), cv2.INTER_LINEAR_EXACT)

    return saliencyMap