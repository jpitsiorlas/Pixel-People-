# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:34:15 2022

@author: Hugo Rechatin
"""

import numpy as np
import cv2
import os
from pathlib import Path
from Data_Aug import load_images_from_folder
            
def check_array(array):
    for i in array:
        if i != 0:
            return 1
    return 0 

def width_extract(image, dest_folder, i):
    W = image.shape[1]
    sample = np.ones((28,28, 4))
    w = 0
    while(w<W):
        if(check_array(image[:,w,3]) == 1):
            first_pixel = w
            while(check_array(image[:,w,3]) == 1):
                w += 1
            pixel_person_width = w - first_pixel
            null_width = 28 - pixel_person_width
            if (pixel_person_width < 28):
                nw_ceil, nw_floor = np.uint8(np.ceil(null_width/2)) , np.uint8(np.floor(null_width/2))
                sample[:,0:nw_ceil,:] = np.zeros((28,nw_ceil,4))
                sample[:,nw_ceil:nw_ceil+pixel_person_width,:] = image[:,first_pixel:w,:]
                sample[:,nw_ceil+pixel_person_width:28,:] = np.zeros((28,nw_floor,4))
            elif(pixel_person_width>=28):
                ew_ceil = np.uint8(np.ceil(-null_width/2))
                ew_floor = np.uint8(np.floor(-null_width/2))
                sample[:,:,:] = image[:,first_pixel+ew_ceil:w-ew_floor,:]
            cv2.imwrite(os.path.join(dest_folder, f"{str(i)}.png"), sample)
            i += 1
            w += 1
        else:
            w += 1
    return i

def height_extract(image):
    H = image.shape[0]
    h = 0
    lines = []
    while(h<H):
        if(check_array(image[h,:,3]) == 1):
            while(check_array(image[h,:,3])==1):
                h+=1
            lines.append((h-28,h))
        else:
            h+=1
    return lines

def extract_samples_from_images(og_folder,dest_folder, i):
    images = load_images_from_folder(og_folder)
    lines = []
    for image in images:
        lines = height_extract(image)
        lines = np.uint32(lines)
        for line in lines:
            print(line[0], line[1])
            i = width_extract(image[line[0]:line[1],:,:], dest_folder, i)
            print (i)

if __name__ == "__main__":
    og_folder = 'Bros/Prepreprocess/images/unprocessed'
    dest_folder = 'Bros/Prepreprocess/samples'
    extract_samples_from_images(og_folder,dest_folder, 6019)
