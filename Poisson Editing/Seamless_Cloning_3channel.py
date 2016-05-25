

import cv2
import numpy as np
from PIL import Image


def process_layer(bs, bt, b, sigma):
    for a in range(it):
        for [i,j] in sigma:
            term = 10000;
            term = term + bs[i+1, j]+bs[i-1, j]+bs[i, j-1]+bs[i, j+1]
            if(im_mask[i-1, j]==255):
                term = term + b[i,j]-b[i-1,j]
            if(im_mask[i+1, j]==255):
                term = term + b[i,j]-b[i+1,j]
            if(im_mask[i, j+1]==255):
                term = term + b[i,j]-b[i,j+1]
            if(im_mask[i, j-1]==255):
                term = term + b[i,j]-b[i,j-1]
            bt[i,j] = (term-10000)/4
        bs = bt.copy()
        print a
    return bs

im_cloned = cv2.imread("sky_cloned.jpg")
im_mask = cv2.imread("sky_mask.jpg",0)

it = 300; # Set number of iterations



im_temp = im_cloned.copy()
im_seamless = im_temp.copy()
sigma = []
for i in range(im_cloned.shape[0]):
        for j in range(im_cloned.shape[1]):
            if (im_mask[i,j]==255):
                sigma.append([i,j])
b , g, r = cv2.split(im_cloned);
bt, gt, rt = cv2.split(im_temp);
bs,gs,rs = cv2.split(im_seamless);

b1 = process_layer(bs,bt,b,sigma);
r1= process_layer(rs,rt,r,sigma);
g1 = process_layer(gs,gt,g,sigma);

bs,rs,gs = b1,r1,g1

rgbArray=np.zeros(im_cloned.shape, 'uint8')
rgbArray[...,0]=rs
rgbArray[...,1]=gs
rgbArray[...,2]=bs
img= Image.fromarray(rgbArray);
img.save("a.jpeg");


