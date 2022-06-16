# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:29:16 2022

@author: 33650
"""

import numpy as np
import torch
import random
import cv2
import os
from  tqdm import tqdm

def load_images_from_folder(folder, tensor = False):
    images = []
    for filename in os.listdir(folder):
        print(folder, filename)
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if tensor == True:
                images.append(torch.from_numpy(img))    
            else:
                images.append(img)
    return images

def save_gray_scale_imgs(images, dest_folder):
    for idx, img in enumerate(images):
        # gs_img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) + (255 - img[:,:,3]))
        gs_img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(os.path.join(dest_folder, f'{idx}.png'), gs_img)

def color_shift(im,idx):
    transparency_layer = im[:,:,3]
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    ycc[:,:,0]
    if(np.uint8(random.uniform(0, 2)) == 0):
       ycc[:,:,1] += np.uint8(np.log(idx))
       ycc[:,:,2] -= np.uint8(np.log(idx))
       ycc[:,:,0] += np.uint8(np.cos(idx))
    else:
        ycc[:,:,1] -= np.uint8(np.log(idx)*idx)
        ycc[:,:,2] += np.uint8(np.log(idx)*idx)
        ycc[:,:,0] -= np.uint8(np.cos(idx))
    brg = cv2.cvtColor(ycc, cv2.COLOR_YCR_CB2BGR)
    out_png = np.empty_like(im)
    out_png[:,:,0:3] = brg
    out_png[:,:,3] = transparency_layer
    return out_png

def augment_data(images, folder):
    i = 0;
    for im in images:
        h_im = cv2.flip(im, 1)
        cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), im)
        i += 1
        cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), h_im)
        i += 1
        idx_list = np.linspace(1, 100, 4)
        for idx in idx_list:
            shifted = color_shift(im, idx)
            h_shifted = color_shift(h_im, idx)
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), shifted)
            i += 1
            cv2.imwrite(os.path.join(folder, f"{str(i)}.png"), h_shifted)
            i += 1
        i+=1
        print(i)
            
def remove_weird_blocks(images):
    for idx, im in tqdm(enumerate(images)):
        im[:,:,0] = np.where(im[:,:,3] == 255, im[:,:,0], 255)
        im[:,:,1] = np.where(im[:,:,3] == 255, im[:,:,1], 255)
        im[:,:,2] = np.where(im[:,:,3] == 255, im[:,:,2], 255)
        cv2.imwrite(f'no_blocks/{idx}.png', im)

def silhouettes(images):
    for idx, im in tqdm(enumerate(images)):
        cv2.imwrite(f'silhouettes/{idx}.png', 255 - im[:,:,3])
        
if __name__ == "__main__":    
    images = load_images_from_folder('silhouette_data/silhouettes')
    
    folder = 'aug_data_no_blocks'
    #augment_data(images, folder)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    