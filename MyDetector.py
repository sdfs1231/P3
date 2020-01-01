import os
import argparse
import torch
import cv2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from data import get_train_test_set,channel_norm
from MyNet import Net
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from generate_own_list import Generate_list
from init import del_imgs,del_txts


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion
    # print(train_loader)
    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        train_mean_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # print(len(batch))
            img = batch['image']
            landmark = batch['landmarks']


            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            # get loss
            loss = pts_criterion(output_pts, target_pts)

            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                # print(output_pts,target_pts)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
                )
        train_losses.append(loss.item())

    ######################
    # validate the model #
    ######################
    valid_mean_pts_loss = 0.0

    model.eval()  # prep model for evaluation without changing the parameters
    with torch.no_grad():
        valid_batch_cnt = 0

        for valid_batch_idx, batch in enumerate(valid_loader):
            valid_batch_cnt += 1
            valid_img = batch['image']
            landmark = batch['landmarks']

            input_img = valid_img.to(device)
            target_pts = landmark.to(device)

            output_pts = model(input_img)

            valid_loss = pts_criterion(output_pts, target_pts)

            valid_mean_pts_loss += valid_loss.item()

            #record per 100 times valid loss
            if valid_batch_idx % args.log_interval == 0:
                valid_losses.append(valid_mean_pts_loss/valid_batch_cnt*1.)

        valid_mean_pts_loss /= valid_batch_cnt * 1.0
        print('Valid: pts_loss: {:.6f}'.format(
            valid_mean_pts_loss
        )
        )
    print('====================================================')
    # save model
    if args.save_model:
        if args.finetune_resnet:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) +'finetune_resnet'+ '.pt')
        else:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')
        torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


def main_test():
    parser = argparse.ArgumentParser(description='MyDetector')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save_directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',  # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--face_box', type=str, default='',nargs='+',
                        help='left top and right bottom cordinates,use space to seperate')
    parser.add_argument('--finetune_resnet', action='store_true', default=False,
                        help='finetuen resnet18 model')
    args = parser.parse_args()
    ###################################################################################
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    print(device)
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    # print(len(train_loader))
    print('===> Building Model')
    # For single GPU
    model = Net().to(device)
    ####################################################################
    criterion_pts = nn.MSELoss()
    cls_criterion_pts = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr )
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        plt.figure(1)
        train_x = np.arange(0,len(train_losses))
        plt.plot(train_x,train_losses,color='r',linestyle='-',label='train loss')
        plt.title('valid loss%f'%valid_losses[-1])
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.savefig(os.path.join('plot','train'+str(args.batch_size)+'_'+str(args.lr)+'.jpg'))
        plt.show()

        print('====================================================')


    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
    # how to do test?
        if not os.path.isfile(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')):
            print('No pre trained file found!')
            return None
        else:
            model.load_state_dict(
                torch.load(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')))
        test_image = None
        test_landmarks = None

        output_pts = model(test_image)
        test_loss = criterion_pts(output_pts, test_landmarks)
        print('Test loss is : {.2f}'.format(test_loss.item()))
        print('====================================================')

    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
    # how to do finetune?
        if not args.finetune_resnet:

            if not os.path.isfile(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')):
                print('No pre trained file found!')
                return None
            #if we only want to change the last layer
            model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')))
            #if we want to change output dim to 100
            model.ip3 = nn.Linear(...,100)
            optimizer = optim.Adam(params=[model.ip3.weight, model.ip3.bias], lr=1e-5)
        else:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(in_features=512,out_features=42)
            optimizer = optim.Adam(params=[model.fc.weight, model.fc.bias], lr=5*1e-5)
        for para in list(model.parameters())[:-1]:
            para.requires_grad = False
        #loss
        criterion_pts = nn.MSELoss()
        #smaller lr

        train_set,test_set = get_train_test_set()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
        finetune_train_losses, finetune_valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        plt.figure(1)
        train_x = np.arange(0, len(finetune_train_losses))
        valid_x = np.arange(0, len(finetune_valid_losses))

        plt.plot(train_x, finetune_train_losses, color='r', linestyle='-', label='train loss')
        plt.title('valid loss:%f'%finetune_valid_losses[-1])
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.savefig(os.path.join('plot','finetune'+str(args.batch_size)+'_'+str(args.lr)+'.jpg'))
        plt.show()
        print('====================================================')

    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')#I\000826.jpg 178 86 566 489
    # how to do predict?
        if not os.path.isfile(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')):
            print('No pre trained file found!')
            return None
        elif not args.face_box:
            print('No face detect!')
            return None
        rect = list(map(int,args.face_box))
        model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs) + '.pt')))
        raw_image =  Image.open('test.jpg')
        predict_image = raw_image.convert('L') # TODO
        # plt.figure()
        # plt.imshow(predict_image)
        # plt.pause(5)
        # print(predict_image.shape())
        crop_w = rect[2]- rect[0] +1
        crop_h = rect[3]- rect[1] +1
        predict_image_crop = predict_image.crop(tuple(rect)) #TODO
        image_resize = np.asarray(
            predict_image_crop.resize((112, 112), ),
            dtype=np.float32)
        # image_resize = np.expand_dims(image_resize, axis=-1)
        # plt.figure()
        # plt.imshow(image_resize)
        # plt.pause(5)
        image_resize = np.expand_dims(image_resize,axis=0)
        image_resize = np.expand_dims(image_resize, axis=0)
        image = torch.from_numpy(image_resize)
        model.eval()
        output_pts = model(image)
        # print(output_pts[0][0].item())
        raw_image = cv2.cvtColor(np.asarray(raw_image), cv2.COLOR_RGB2BGR)
        for idx in range(0,output_pts.size()[1],2):
            x = output_pts[0][idx].item()
            y = output_pts[0][idx+1].item()

            cv2.circle(raw_image,(int(x*crop_w/112+rect[0]),int(y*crop_h/112+rect[1])),1,(0,0,255),4)
        cv2.imshow('predict',np.asarray(raw_image))
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
        print('====================================================')


if __name__ == '__main__':
    print("==> Start Cleaning the environment")
    del_imgs()
    del_txts()
    print("="*5,'Done','='*5)

    print('==> Generating the data set')
    test = Generate_list('label.txt', aug=True)
    test.seperate_write_file()
    print('='*5,'done','='*5)

    main_test()