import cv2
import os
import random

#crop dataset only include names I/image_name.jpg

class Generate_Non_Face():
    def __init__(self,dataset,crop_dataset,debug=False,crop_nums=1,iou_ratio=0.2):
        self.dataset = dataset
        self.crop_datase = crop_dataset
        self.debug = debug
        self.crop_nums = crop_nums
        self.iou_ratio = iou_ratio

    def random_crop(self):
        crops = []
        for img_name in self.crop_datase:
            img = cv2.imread(os.path.join('data',img_name))
            h,w = img.shape[:2]
            left = []
            top = []
            right = []
            bottom = []
            for info in self.dataset[img_name]:
                left.append(info[0][0])
                top.append(info[0][1])
                right.append(info[0][2])
                bottom.append(info[0][3])
            orgleft = min(left)
            orgtop = min(top)
            orgright = max(right)
            orgbottom = max(bottom)
            orgrect = [orgleft,orgtop,orgright,orgbottom]
            while len(crops)<self.crop_nums:
                cropleft = random.uniform(1,int(w/2))
                croptop = random.uniform(1,int(h/2))
                cropright = random.uniform(cropleft,int(w))
                cropbottom = random.uniform(croptop,int(h))
                croprect = [cropleft,croptop,cropright,cropbottom]
                if self.check_iou(orgrect,croprect)<self.iou_ratio:
                    self.dataset[img_name].append((croprect,0))
                    crops.append((img_name,croprect))
                    if self.debug:
                        self.check(img_name,croprect)

    def check_iou(self,rect1,rect2):
        inleft = max(rect1[0],rect2[0])
        intop = max(rect1[1],rect2[1])
        inright = min(rect1[2],rect2[2])
        inbottom = min(rect1[3],rect2[3])
        if inright<inleft or inbottom<intop:
            innerarea = 0.
        else:
            innerarea = float((inright-inleft+1)*(inbottom-intop+1))
        area1 = float((rect1[2]-rect1[0]+1)*(rect1[3]-rect1[1]+1))
        area2 = float((rect2[2]-rect2[0]+1)*(rect2[3]-rect2[1]+1))
        iou = float(innerarea/(area1+area2-innerarea))
        return iou

    def check(self,img_name,rect):
        img = cv2.imread(os.path.join('data',img_name))
        img_crop = img[rect[0]:rect[2],rect[1]:rect[3]]
        cv2.imshow('crop',img_crop)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
if __name__=='__main__':
    testdataset= ['I\\000096.jpg']
    infos = {'I\\000096.jpg':[([1,2,3,4],[(1,2),(3,4)])]}
    # print(infos['I\\000007.jpg'][0][1])
    test = Generate_Non_Face(testdataset,infos,debug=True)
    img = test.random_crop()