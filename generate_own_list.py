#According lable.txt draw the bbox and landmarks of img
#Recording the changes of landmarks and bboxes
#Save as train or test

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import re
from Augment import Augment
from Generate_Non_Face import Generate_Non_Face

class Generate_list():
    def __init__(self,labelfile,aug=False):
        self.no_face_ratio = 0.05
        self.aug = aug
        self.aug_ratio = 0.1
        self.foldername = os.path.join('data')
        self.part = ['I','II']
        self.labelfile = labelfile
        self.save_file = ['train.txt','test.txt','val.txt']
        self.val_percent = 0
        self.test_percent = 0.1
        self.train_percent = 0.9
        self.expand_roi_ratio = 0.25
        self.rawDataset ,self.roi_expand_Dataset = self.generate_Dataset()




    def process_paras(self,imgs,part,lines):#Get original recs and landmarks cords
        for line in lines:
            line = line.strip().split()
            name = os.path.join(part,line[0])
            if name not in imgs.keys():
                imgs[name] = []
            # else:
                # print('photo %s has more than 1 face'%name)

            rect = list(map(int, list(map(float, line[1:5]))))
            x = list(map(float, line[5:len(line):2]))
            y = list(map(float, line[6:len(line):2]))
            landmarks = list(zip(x, y))
            imgs[name].append((rect, landmarks,1))
        # print(temp)
        return imgs

    def indentify_unlable(self,part,lines):#  find unlabeled imgs
        labeledimgs = {}
        for line in lines:
            detect = line.split()[0]
            p = os.path.isfile(os.path.join(self.foldername,part,detect))
            if not p:
                print('For folder %s, image %s is unlabeled!!!'%part%detect)
            else:
                labeledimgs[os.path.join(part,detect)] = []
        return labeledimgs  #  part/image name.jpg


    def generate_Dataset(self): #Load original label.txt to find out unlabeled imgs and generate Dataset
        imgs = {} #imgs = {name:[ [rec 4 cords,x1,y1,x2,y2] ,[landmarks (x,y),...] ]}
        roi_imgs = {} #format same as imgs above

        for part in self.part: #Generate raw data
            labelfile = os.path.join(self.foldername,part,self.labelfile)
            if not os.path.isfile(labelfile): #Can also write as try:...expect:...
                print('label file is can\'t be found!')
                return None,None
            else:
                with open(labelfile) as f:
                    labledimgs = f.readlines()
                self.indentify_unlable(part,labledimgs)
                imgs = self.process_paras(imgs,part,labledimgs)

        if self.aug:
            nums_aug = int(len(imgs.keys()) * self.aug_ratio)
            aug_imgs = random.sample(imgs.keys(), nums_aug)
            aug_imgs = [x for x in aug_imgs if re.match(r'.*\d{6}\.jpg', x)]
        #augmentation
            aug = Augment(aug_imgs,imgs)
            aug.color_change()

            nums_aug = int(len(imgs.keys()) * self.aug_ratio)
            aug_imgs = random.sample(imgs.keys(), nums_aug)
            aug_imgs = [x for x in aug_imgs if re.match(r'.*\d{6}\.jpg', x)]
            aug = Augment(aug_imgs, imgs)
            aug.bright_change()
            print('='*20,'Augmentation Done','='*20)
            # for name in aug_imgs:
            #
            #     org_img = cv2.imread(os.path.join('data',name))
            #     h, w = org_img.shape[:2]
            #     flip_img = cv2.flip(org_img, 1)
            #     flip_img_name = name[:-4]+'flip'+'.jpg'
            #     rect = imgs[name][-1][0]
            #     rect_w = rect[2] - rect[0] + 1
            #     landmarks = imgs[name][-1][1]
            #     new_rect = [w - rect[0] -rect_w, rect[1], w - rect[2] +rect_w, rect[3]]
            #     new_landmarks = []
            #     for l in landmarks:
            #         new_landmarks.append((w - l[0] + 1, l[1]))
            #
            #     cv2.imwrite(os.path.join('data',flip_img_name) , flip_img)
            #
            #     imgs[flip_img_name] = []
            #     imgs[flip_img_name].append((new_rect, new_landmarks))

        # Generate roi expand data
        for img in imgs:
            # print(img)
            if img not in roi_imgs:
                roi_imgs[img] = []
            info = imgs[img]
            new_info = self.expand_roi(img,info,self.expand_roi_ratio)
            roi_imgs[img] = new_info
        print('=' * 20, 'Expand Roi Done', '=' * 20)


        # Generate None Face data
        nums_no_face = int(len(roi_imgs.keys()) * self.no_face_ratio)
        no_face_candidate = random.sample(roi_imgs.keys(),nums_no_face)
        no_face_candidate = [x for x in no_face_candidate if re.match(r'.*\d{6}\.jpg', x)]
        # print(no_face_candidate)
        no_face_generator = Generate_Non_Face(no_face_candidate,roi_imgs)
        no_face_generator.random_crop()
        print('=' * 20, 'Generate No-Face Done', '=' * 20)

        return imgs, roi_imgs




    def expand_roi(self,img_name,img_info,ratio=0.25): #Per image input all paras update (consider to concate name and info)
        file = os.path.join(self.foldername, img_name)
        img = cv2.imread(file)
        img_h, img_w, _ = img.shape
        roi_img = []
        for info in img_info:
            box = info[0]
            landmarks = info[1]
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            width = abs(x2 - x1 + 1)
            height = abs(y2 - y1 +1)
            wpad = width * ratio
            hpad = height * ratio
            roi_x1 = x1 - wpad
            roi_y1 = y1 - hpad
            roi_x2 = x2 + wpad
            roi_y2 = y2 + hpad
            roi_x1 = 0 if roi_x1<0 else roi_x1
            roi_y1 = 0 if roi_y1<0 else roi_y1
            roi_x2 = img_w - 1 if roi_x2>img_w else roi_x2
            roi_y2 = img_h - 1 if roi_y2>img_h else roi_y2
            new_rec = list(map(int,[roi_x1,roi_y1,roi_x2,roi_y2]))
            new_landmarks = landmarks - np.array([roi_x1,roi_y1])
            new_landmarks[new_landmarks<0] = 0
            new_landmarks = new_landmarks.tolist() #change to list in order to write
            roi_img.append((new_rec,new_landmarks,1))
        # if img_name == 'II\\008520.jpg':
        #     print(len(roi_img))
        return roi_img

    def write_file(self,filename,data):
        #Seperation
        if os.path.isfile(os.path.join(self.foldername,filename)):
            # Because file open as 'a', dont want to append
            # exit file anymore!
            print('%s has exited,removing......'%filename)
            os.remove(os.path.join(self.foldername,filename))
            print('='*10+'Removing Done'+'='*10)
        with open(os.path.join(self.foldername,filename),'a') as f:
            for name in data:
                infos = self.roi_expand_Dataset[name]
                # if len(infos)>2:
                #     # print(name)
                for info in infos:
                    s = name + ' ' + str(info).replace(' ', '').replace('(', '').replace('[', '')\
                        .replace(',', ' ').replace(')','').replace(']', '')
                    f.write(s+'\n')

    def seperate_write_file(self):
        # train_data = []
        # test_data = []
        val_data = []
        nums = len(self.roi_expand_Dataset)
        names = list(self.roi_expand_Dataset.keys())
        random.shuffle(names)
        cut = int(nums*self.train_percent)
        train_data = names[:cut]
        test_data = names[cut:]
        # for i,name in enumerate(self.roi_expand_Dataset):
        #         #     if i<=self.test_percent*nums or len(test_data)< self.test_percent*nums:
        #         #         toll = random.choice([train_data,test_data])
        #         #         toll.append(name)
        #         #     else:
        #         #         train_data.append(name)

            # if i<=self.train_percent*nums:
            #     train_data.append(name)
            # else:
            #     if i>(self.train_percent+self.test_percent)*nums and self.val_percent!=0:
            #         val_data.append(name)
            #     else:
            #         test_data.append(name)
        self.write_file(self.save_file[0],train_data)
        self.write_file(self.save_file[1], test_data)
        self.write_file(self.save_file[2], val_data)

    def check(self): #Random choose a pic to draw the box
        trail = random.choice(list(self.roi_expand_Dataset.keys()))
        while not re.match('.*flip.jpg',trail):
            trail = random.choice(list(self.roi_expand_Dataset.keys()))
        img = cv2.imread(os.path.join(self.foldername, trail))

        rec_raw = self.rawDataset[trail][0][0]
        rec_roi = self.roi_expand_Dataset[trail][0][0]

        # landmarks_raw = self.rawDataset[trail][0][1]
        landmarks_roi = self.roi_expand_Dataset[trail][0][1]

        draw1 = cv2.rectangle(img, (rec_raw[0], rec_raw[1]), (rec_raw[2], rec_raw[3]), (0, 0, 255), 2)
        draw2 = cv2.rectangle(img,(rec_roi[0],rec_roi[1]),(rec_roi[2],rec_roi[3]),(0,0,255),2)

        for point in landmarks_roi:
            cv2.circle(img,(int(point[0]+rec_roi[0]),int(point[1])+rec_roi[1]),1,(0,0,255))
        cv2.imshow('test',img)
        cv2.waitKey(0)
        cv2.destroyWindow("test")

        # for point in landmarks_raw:
        #     cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255))
        # cv2.imshow('raw',draw1)
        # cv2.waitKey(0)
        # cv2.destroyWindow('raw')

if __name__=='__main__':
    from init import del_txts,del_imgs
    del_txts()
    del_imgs()
    test = Generate_list('label.txt',aug=True)
    test.seperate_write_file()