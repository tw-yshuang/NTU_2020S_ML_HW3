import torchvision.transforms as transforms
import torch.nn as nn
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from data_process import ImgDataset, DataPreProcess
from net import Net_Classifier
from training_model import HW3_Model


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device State:', device)

    return device


def read_pickle(path):
    with open(path, 'rb') as target:
        return pickle.load(target)


if __name__ == "__main__":
    img_inChannel = 3
    # train = DataPreProcess('Data/food-11/training', 128,
    #                        img_inChannel, savePath='Data/train.pickle')
    # val = DataPreProcess('Data/food-11/validation', 128,
    #                      img_inChannel, savePath='Data/val.pickle')
    # test = DataPreProcess('Data/food-11/testing', 128,
    #                       img_inChannel, isTest=True,  savePath='Data/test.pickle')
    train = read_pickle('Data/train.pickle')
    val = read_pickle('Data/val.pickle')
    test = read_pickle('Data/test.pickle')

    for i, dataset in enumerate([train.x, val.x, test.x]):
        names = ['train', 'validation', 'test']
        print('Size of {} is {}'.format(names[i], len(dataset)))

    # training use agumentation
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # degree of rotat e.g. 15=(-15, 15)
        transforms.ToTensor(),  # data normalization
    ])
    # testing dosen't use agumentation
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # data normalization
    ])

    train_set = ImgDataset(train.x, train.y, train_transforms)
    val_set = ImgDataset(val.x, val.y, test_transforms)

    # upload to DataLoader
    BATCH_SIZE = 128
    NUM_WORKERS = 6
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS)

    # training
    device = get_device()
    net = Net_Classifier(img_inChannel).to(device)
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    EPOCH = 30

    model = HW3_Model(device, net, loss, optimizer)
    model.training(train_loader, val_loader, NUM_EPOCH=30,
                   saveDir='./out', checkpoint=2)
