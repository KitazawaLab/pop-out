import numpy as np
import cv2
from matplotlib import pyplot as plt

def spectral_residual_saliency(img, WIDTH=64):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ratio = int(img.shape[0]/img.shape[1])
    img_n = cv2.resize(img, (WIDTH,WIDTH*ratio), cv2.INTER_LINEAR)
    
    c = cv2.dft(np.float32(img_n), flags = cv2.DFT_COMPLEX_OUTPUT)
    amp = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)
    spectralResidual = np.exp(np.log(1+amp) - cv2.boxFilter(np.log(1+amp), -1, (3,3),cv2.BORDER_DEFAULT))
    
    c[:,:,0] = c[:,:,0] * spectralResidual / amp # phase is preserved
    c[:,:,1] = c[:,:,1] * spectralResidual / amp # phase is preserved
    c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
    amp = c[:,:,0]**2 + c[:,:,1]**2
    amp = cv2.normalize(cv2.GaussianBlur(amp,(5,5),8,0,cv2.BORDER_DEFAULT), 0., 1., cv2.NORM_MINMAX)
    amp = cv2.resize(amp, (img.shape[0],img.shape[1]), 0, 0, cv2.INTER_LINEAR)
    
    return amp