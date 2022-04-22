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
            if (pixel_person_width%2 == 0):
                sample[:,:,:] = image[:,first_pixel-np.uint8(null_width/2) : (w)+np.uint8(null_width/2),:]
            else:
                sample[:,:,:] = image[:,first_pixel-np.uint8(np.ceil(null_width/2)) : (w)+np.uint8(np.floor(null_width/2)),:]
            cv2.imwrite(os.path.join(dest_folder, f"{str(i)}.png"), sample)
            i += 1
            print(i)
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

def extract_samples_from_images(og_folder,dest_folder):
    images = load_images_from_folder(og_folder)
    i = 0
    lines = []
    for image in images:
        lines = height_extract(image)
        lines = np.uint16(lines)
        for line in lines:
            print(line[0], line[1])
            i = width_extract(image[line[0]:line[1],:,:], dest_folder, i)

if __name__ == "__main__":
    og_folder = 'Bros/Prepreprocess/images'
    dest_folder = 'Bros/Prepreprocess/samples'
    extract_samples_from_images(og_folder,dest_folder)
