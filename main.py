import torchvision.transforms as transforms
import torch.nn as nn
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from src.data_process import ImgDataset, DataPreProcess
from src.net import Net_Classifier, Net_Classifier_cyc, Net_Classifier_mass
from src.training_model import HW3_Model


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device State:', device)

    return device


def read_pickle(path):
    with open(path, 'rb') as target:
        return pickle.load(target)


def training_analysis():
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
    train_transforms_arg = transforms.Compose([
        transforms.ColorJitter(brightness=(
            0, 50), contrast=(0, 25), saturation=(0, 10)),  # hue=(-0.25, 0.25)
        # degree of rotat e.g. 15=(-15, 15)
        transforms.RandomRotation(50, expand=False),
        transforms.Resize((128, 128)),
    ])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([train_transforms_arg], p=0.9),
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
    NUM_WORKERS = 3  # depand on transforms step
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS)

    # training
    device = get_device()
    # net is use ntutiem_cyc server
    net = Net_Classifier_cyc(img_inChannel).to(device)
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 4 :0.62
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    model = HW3_Model(device, net, loss, optimizer)
    model.training(train_loader, val_loader, NUM_EPOCH=10,
                   saveDir='./out', checkpoint=50, bestModelSave=True)
    model.get_performance_plt()


def best_training():
    img_inChannel = 3
    train = read_pickle('Data/train.pickle')
    val = read_pickle('Data/val.pickle')
    test = read_pickle('Data/test.pickle')

    for i, dataset in enumerate([train.x, val.x, test.x]):
        names = ['train', 'validation', 'test']
        print('Size of {} is {}'.format(names[i], len(dataset)))

    # training use agumentation
    train_transforms_arg = transforms.Compose([
        transforms.ColorJitter(brightness=(
            0, 50), contrast=(0, 25), saturation=(0, 10)),  # hue=(-0.25, 0.25)
        # degree of rotat e.g. 15=(-15, 15)
        transforms.RandomRotation(25, expand=True),
        transforms.Resize((128, 128)),
    ])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([train_transforms_arg], p=0.85),
        transforms.ToTensor(),  # data normalization
    ])

    train.x = np.concatenate((train.x, val.x), axis=0)
    train.y = np.concatenate((train.y, val.y), axis=0)
    train_set = ImgDataset(train.x, train.y, train_transforms)

    # upload to DataLoader
    BATCH_SIZE = 128
    NUM_WORKERS = 3  # depand on transforms step
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    # training
    device = get_device()
    # net is use ntutiem_cyc server
    net = Net_Classifier_cyc(img_inChannel).to(device)
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 4 :0.62
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    model = HW3_Model(device, net, loss, optimizer)
    model.training(train_loader, NUM_EPOCH=200,
                   saveDir='./out', checkpoint=50, bestModelSave=True)
    model.get_performance_plt()


def pre_training(path):
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
    train_transforms_arg = transforms.Compose([
        transforms.ColorJitter(brightness=(
            0, 50), contrast=(0, 25), saturation=(0, 10)),  # hue=(-0.25, 0.25)
        # degree of rotat e.g. 15=(-15, 15)
        transforms.RandomRotation(50, expand=False),
        transforms.Resize((128, 128)),
    ])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([train_transforms_arg], p=0.9),
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
    NUM_WORKERS = 3  # depand on transforms step
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS)

    device = get_device()
    # net is use ntutiem_cyc server
    net = Net_Classifier_cyc(img_inChannel).to(device)
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 4 :0.62
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    model = HW3_Model(device, net, loss, optimizer)
    model.load_model(path)
    model.training(train_loader, val_loader, NUM_EPOCH=50,
                   checkpoint=50, bestModelSave=True)
    model.get_performance_plt()


if __name__ == "__main__":
    training_analysis()
    # pre_training('out/0819-1355/best_e010_0.0125.pickle')
    # best_training()
