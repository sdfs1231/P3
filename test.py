import cv2
from PIL import Image
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt
print(5e-5-0.00005)
print(round(random.uniform(0,3),1))



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
# org_img = cv2.imread(os.path.join(choose,pickname[0]))
# print(org_img.shape)
# flip_img = cv2.flip(org_img,1)
# cv2.imshow('test',flip_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
# print(re.match(r'\w+\.jpg',pickname[0]))
# print(len(pickname))
