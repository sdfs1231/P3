import numpy as np
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools


folder_list = ['I', 'II']
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)-1]))
    # if cls !=0 or cls !=1:
    #     print(img_name)
    cls = [float(line_parts[-1])]
    return img_name, rect, landmarks,cls


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks, cls= sample['image'], sample['landmarks'],sample['class']
        image_resize = np.asarray(
                            cv2.resize(image,(train_boarder, train_boarder)),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        cls = np.array(cls).astype(np.float32)
        return {'image': image,
                'landmarks': landmarks,
                'class': cls
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks,cls = sample['image'], sample['landmarks'],sample['class']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, axis=0)

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'class': torch.from_numpy(cls).long()}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase,transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks,cls= parse_line(self.lines[idx])
        # image cv2
        img = cv2.imread(os.path.join('data',img_name))
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_crop = img[rect[1]:rect[3],rect[0]:rect[2]]
        # image PIL Image
        # img = Image.open(os.path.join('data',img_name)).convert('L')#convert to single channel
        # img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)
        # print(landmarks.shape)
        if not landmarks.shape[0]:
            landmarks = np.zeros((42,)).astype(np.float32)

		
		
        # you should let your landmarks fit to the train_boarder(112)
		# please complete your code under this blank
		# your code:
        w = rect[2] - rect[0]+1
        h = rect[3] - rect[1]+1
        #train_border is the same
        w_ratio = train_boarder/w
        h_ratio = train_boarder/h
        for idx in range(0,landmarks.size,2):
            landmarks[idx] *= w_ratio
            landmarks[idx+1] *= h_ratio
        sample = {'image': img_crop, 'landmarks': landmarks,'class':cls}
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    with open(os.path.join('data',data_file)) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase,transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    print(len(train_loader.dataset))
    for batch_idx, batch in enumerate(train_loader):
        img = batch['image']
        landmark = batch['landmarks']
        cls = batch['class']
        # print(batch['class'].shape)
        # print(batch)

        # if i['class'] == 0:
        #     print(i)
        #     break
    # for i in range(1, len(train_set)):
    #     sample = train_set[i]
    #     img = sample['image']
    #     landmarks = sample['landmarks']
    #     cls = sample['class']
    #     # print(landmarks.size)
    #     ## 请画出人脸crop以及对应的landmarks
	# 	# please complete your code under this blank
    #     # print(img.size())
    #     img = img.squeeze(0)
    #     # print(img.size())
    #     img = img.numpy()
    #     # img = np.asarray(img,dtype=np.int32)
    #     for idx in range(0,landmarks.size,2):
    #         cv2.circle(img,(int(landmarks[idx].item()),int(landmarks[idx+1].item())),1,(0,0,255),8)
    #     cv2.imshow('face' if cls.item() else 'non_face',img)
    #     key = cv2.waitKey()
    #     if key == 27:
    #         exit(0)
    #     cv2.destroyAllWindows()







