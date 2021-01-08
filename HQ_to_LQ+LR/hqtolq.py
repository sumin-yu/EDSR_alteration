import cv2
import numpy as np
from random import random
import copy

def salt_and_pepper(image, p, s): 
        output = np.zeros(image.shape)
        for i in range (image.shape[0]):
                for j in range(image.shape[1]):
                        rdn1 = random()
                        rdn2 = random()
                        #if rdn1 > p or rdn2 > s:
                               #output[i][j] = image[i][j]
                        #elif np.all(image[i][j]) > 128:
                               #if rdn1 < p :
                                     #output[i][j] = image[i][j]-100
                        #elif np.all(image[i][j]) <= 128:
                               #if rdn2 > s:
                                     #output[i][j] = image[i][j] + 100
                        if rdn1 < p:                        
                                output[i][j] = 10 
                        elif rdn2 < s:
                                output[i][j] = 200 
                        else:
                                output[i][j] = image[i][j]
        return output

def noisy(noise_type, image):
   if noise_type == "gauss":
      row,col,ch = image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean, sigma, (row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_type == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image*vals)/float(vals)
      return noisy
   elif noise_type == "speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

img = 1
for i in range(1,10):
	print('imgae num {} is now changing'.format(img))
	src = cv2.imread('/data3/smyoo/DIV2K/DIV2K_train_HR/000{}.png'.format(i), cv2.IMREAD_COLOR)
	dst = cv2.blur(src,(9,9))
	dst = salt_and_pepper(dst,0.001,0.001)
	dst = cv2.blur(dst, (5,5))
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LQ/000{}x1.png'.format(i), dst)
	LQ  = cv2.resize(dst, dsize=(0,0), fx=0.5,fy=0.5, interpolation = cv2.INTER_LINEAR)
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LR/000{}x2.png'.format(i), LQ)
	img = img +1

img = 10
for i in range(10,100):
	print('imgae num {} is now changing'.format(img))
	src = cv2.imread('/data3/smyoo/DIV2K/DIV2K_train_HR/00{}.png'.format(i), cv2.IMREAD_COLOR)
	dst = cv2.blur(src, (9,9))
	dst = salt_and_pepper(dst,0.001,0.001)
	dst = cv2.blur(dst, (5,5))
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LQ/00{}x1.png'.format(i), dst)
	LQ  = cv2.resize(dst, dsize=(0,0), fx=0.5,fy=0.5, interpolation = cv2.INTER_LINEAR)
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LR/00{}x2.png'.format(i), LQ)
	img = img +1

img = 100
for i in range(100,901):
	print('imgae num {} is now changing'.format(img))
	src = cv2.imread('/data3/smyoo/DIV2K/DIV2K_train_HR/0{}.png'.format(i), cv2.IMREAD_COLOR)
	dst = cv2.blur(src, (9,9))
	dst = salt_and_pepper(dst,0.001,0.001)
	dst = cv2.blur(dst, (5,5))
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LQ/0{}x1.png'.format(i), dst)
	LQ  = cv2.resize(dst, dsize=(0,0), fx=0.5,fy=0.5, interpolation = cv2.INTER_LINEAR)
	cv2.imwrite('/data3/smyoo/DIV2K/DIV2K_train_LR_bicubic/LR/0{}x2.png'.format(i), LQ)
	img = img +1




