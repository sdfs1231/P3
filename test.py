import cv2
from PIL import Image
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt
import torch
from MaskedMSELoss import MaskedMSELoss
from MyNet2 import Net
import torch.nn as nn

for i in range(10):
    print(i)
    for j in range(5):
        if j>5:
            break
# a = np.array([[1,2],[3,4],[5,6]])
# b = np.array([[1],[0],[1]])
# a = torch.from_numpy(a)
# b = torch.from_numpy(b)
# print(a*b)
# print(torch.sum(b))
# print(23**2-23)
# def l2(a,b):
#     return (a-b)**2
#
# a = [[1,2,3,4],[9,10,11,12]]
# # b = [[5,6,7,8],[]]
# mask = [1,0]
# a = np.asarray(a,dtype=np.float32)
# # b = np.asarray(b,dtype=np.float32)
# a = torch.from_numpy(a)
# # b = torch.from_numpy(b)
#
# print(a.shape)
# print(a.shape[0])
# mask = np.asarray(mask,dtype=np.int8)
#
# mask = torch.from_numpy(mask)
# mask = torch.reshape(mask,(a.shape[0],1))
# print(a*mask)
# f = nn.MSELoss()
# print(f(a*mask,b*mask))


# a={'a':[],'b':[],'c':[],'d':[]}
# temp = list(a.keys())
# print(temp)
#
# random.shuffle(temp)
# print(temp)
# exit()
# for i in range(100):
#     t = random.choice([a,b])
#     t.append(i)
# print(a)
# print(b)
# a = 'I\\025896flip.jpg'
# if re.match('.*\d{6}\.jpg',a):
#     print('yes')


#
# a = 'II\\0522589.jpg'
# print(a[:-4
#       ])
#
#
#
#
#
# exit()
# if re.match(r'\w{6}\.jpg','020258.jpg'):
#     print('haha')
# exit()
# choose1or2 = ['I','II']
# child_choose = random.choice(choose1or2)
# choose = os.path.join('data',child_choose)
# pathDir = os.listdir(choose)
# # print(pathDir)
# rate = 0.1
# picknum = int((len(pathDir)-1)*0.1)
# pickname = random.sample(pathDir,picknum)
# pickname = [ x for x in pickname if re.match(r'\w{6}\.jpg',x)]

# org_img = cv2.imread(os.path.join('data','I\\004470.jpg'))
# crop_img = org_img[550:584,116:363]
# cv2.imshow('test',crop_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
# print(re.match(r'\w+\.jpg',pickname[0]))
# print(len(pickname))
