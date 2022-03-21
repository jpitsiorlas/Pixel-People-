# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:29:16 2022

@author: 33650
"""

import numpy as np
import random
import cv2
import os
from pathlib import Path

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(folder, filename)
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images

def color_sub(im):
    elim = np.uint8(random.uniform(0, 2))
    out = im
    out[:,:,elim] = 0
    return out
def color_shift(im):
    if(np.uint0(random.uniform(0, 2)) == 0):
        shiftb = np.uint8(random.uniform(0, 256))
        shiftr = (np.uint8(random.uniform(0, 256)) + shiftb)/2
        shiftg = (np.uint8(random.uniform(0, 256)) + shiftb + shiftr)/3
    else:
        shiftb = np.uint8(random.uniform(0, 256))
        shiftr = (np.uint8(random.uniform(0, 256)) + 3*shiftb)/2
        shiftg = (np.uint8(random.uniform(0, 256)) + 2*shiftb + 2*shiftr)/3
    for k in range(0, 27):
        for l in range(0,27):
            if (im[k,l,0]==0):
                im[k,l,0] = 0
            elif(im[k,l,0]+shiftb>255):
                im[k,l,0] = shiftb + im[k,l,0] - 255
            else:
                im[k,l,0] += shiftb
                
            if (im[k,l,1]==0):
                im[k,l,1] = 0
            elif(im[k,l,1]+shiftr>255):
                im[k,l,1] = shiftr + im[k,l,1] - 255
            else:
                im[k,l,1] += shiftr
                
            if (im[k,l,2]==0):
                im[k,l,2] = 0
            elif(im[k,l,2]+shiftg>255):
                im[k,l,2] = shiftg + im[k,l,2] - 255
            else:
                im[k,l,2] += shiftg
    return im

def augment_data(images, folder):
    i = 0;
    for im in images:
        h_im = cv2.flip(im, 1)
        cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), im)
        i += 1
        cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), h_im)
        for j in range(10):
            shifted = color_shift(im)
            h_shifted = color_shift(h_im)
            i += 1
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), shifted)
            i += 1
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), h_shifted)
            i += 1
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), color_sub(shifted))
            i += 1
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), color_sub(h_shifted))
        i+=1
        print(i)

        
if __name__ == "__main__":    
    images = load_images_from_folder('hand made data')
    augment_data(images, 'augmented data')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    