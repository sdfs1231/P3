import cv2
import random
import os
import re
import copy
import numpy as np
class Augment():
    # Dataset must be list including img name that should be augmented
    # Infos must be dict{'imagename':[(rect,landmarks)]}
    def __init__(self,dataset,infos,debug=False):
        self.dataset = dataset
        self.infos = infos
        self.debug = debug


    def flip(self):
        for img_name in self.dataset:
            org_img = cv2.imread(os.path.join('data', img_name))
            h, w = org_img.shape[:2]
            flip_img = cv2.flip(org_img, 1)
            flip_img_name = img_name[:-4] + 'flip' + '.jpg'
            rect = self.infos[img_name][-1][0]
            rect_w = rect[2] - rect[0] + 1
            landmarks = self.infos[img_name][-1][1]
            new_rect = [w - rect[0] - rect_w, rect[1], w - rect[2] + rect_w, rect[3]]
            new_landmarks = []
            for l in landmarks:
                new_landmarks.append((w - l[0] + 1, l[1]))

            cv2.imwrite(os.path.join('data', flip_img_name), flip_img)

            self.infos[flip_img_name] = []
            self.infos[flip_img_name].append((new_rect, new_landmarks,1))
        if self.debug:
            #print augmentation img name
            print(x for x in self.dataset)
        return self.infos

    def color_change(self):
        for img_name in self.dataset:
            img = cv2.imread(os.path.join('data', img_name))
            (h,w,c) = img.shape
            blank = np.zeros([h,w,c],img.dtype)


            new_img = copy.deepcopy(img)
            new_img[:,:,0] = 255 - img[:,:,0]

            new_img[:, :, 1] = 255 - img[:, :, 1]
            new_img[:, :, 2] = 255 - img[:, :, 2]
            ccimg = img_name[:-4] + 'cc' + '.jpg'
            if self.debug:
                cv2.imshow('color change',new_img)
                key = cv2.waitKey()
                if key == 27:
                    cv2.destroyAllWindows()
            else:
                cv2.imwrite(os.path.join('data', ccimg),new_img)
            self.infos[ccimg] = []
            for info in self.infos[img_name]:
                self.infos[ccimg].append((info[0], info[1],1))


    def bright_change(self):
        for img_name in self.dataset:
            c = round(random.uniform(0,5),2)
            b = random.randrange(0,5)
            img = cv2.imread(os.path.join('data', img_name))
            (h, w, c) = img.shape
            blank = np.zeros([h,w,c],img.dtype)
            dst = cv2.addWeighted(img, c, blank, 1-c, b)
            newimg_name = img_name[:-4]+'bc'+'.jpg'
            if self.debug:
                cv2.imshow('bright change',dst)
                key = cv2.waitKey()
                if key == 27:
                    cv2.destroyAllWindows()
            else:
                cv2.imwrite(os.path.join('data', newimg_name),dst)
            self.infos[newimg_name] = []
            for info in self.infos[img_name]:
                self.infos[newimg_name].append((info[0], info[1],1))



if __name__=='__main__':
    testdataset= ['I\\000096.jpg']
    infos = {'I\\000096.jpg':[([1,2,3,4],[(1,2),(3,4)])]}
    # print(infos['I\\000007.jpg'][0][1])
    test = Augment(testdataset,infos,debug=True)
    img = test.color_change()
    print(infos)