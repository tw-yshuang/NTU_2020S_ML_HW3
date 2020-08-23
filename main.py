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


def training_analysis(NUM_EPOCH, valiadating=True):
    # training use agumentation
    train_transforms_arg = transforms.Compose([
        transforms.ColorJitter(brightness=(
            0, 50), contrast=(0, 25), saturation=(0, 10), hue=(-0.25, 0.25)),  # , saturation=(0, 10), hue=(-0.25, 0.25)
        # degree of rotat e.g. 15=(-15, 15)
        transforms.RandomRotation(50, expand=False),
        # transforms.Resize((128, 128)),
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

    # upload to DataLoader
    BATCH_SIZE = 64  # 　<<<<<　Batch size >>>>>
    NUM_WORKERS = 4  # depand on transforms step

    # training
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    model = HW3_Model(device, net, loss, optimizer)

    if valiadating:
        train_set = ImgDataset(
            train.x, train.y, train_transforms)  # 　<<<<<　train_transforms >>>>>
        val_set = ImgDataset(val.x, val.y, test_transforms)
        train_loader = DataLoader(train_set, BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS)
        model.training(train_loader, val_loader, NUM_EPOCH=NUM_EPOCH,
                       saveDir='./out', checkpoint=20, bestModelSave=True)
        model.get_performance_plt()
    else:
        train_set = ImgDataset(np.concatenate(
            (train.x, val.x)), np.concatenate((train.y, val.y)), train_transforms)
        train_loader = DataLoader(train_set, BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        model.training(train_loader, NUM_EPOCH=NUM_EPOCH,
                       saveDir='./out', checkpoint=20, bestModelSave=True)


def pre_training(path, NUM_EPOCH):
    # training use agumentation
    train_transforms_arg = transforms.Compose([
        transforms.ColorJitter(brightness=(
            0, 50), contrast=(0, 25), saturation=(0, 10), hue=(-0.25, 0.25)),  # , saturation=(0, 10), hue=(-0.25, 0.25)
        # degree of rotat e.g. 15=(-15, 15)
        transforms.RandomRotation(50, expand=False),
        # transforms.Resize((128, 128)),
    ])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([train_transforms_arg], p=0.95),
        transforms.ToTensor(),  # data normalization
    ])
    # testing dosen't use agumentation
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # data normalization
    ])

    # 　<<<<< train_transforms >>>>>
    train_set = ImgDataset(train.x, train.y, train_transforms)
    val_set = ImgDataset(val.x, val.y, test_transforms)

    # upload to DataLoader
    BATCH_SIZE = 64  # 　<<<<< Batch size >>>>>
    NUM_WORKERS = 3  # depand on transforms step
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS)

    # training
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 4 :0.62
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    model = HW3_Model(device, net, loss, optimizer)
    model.load_model(path)
    model.training(train_loader, val_loader, NUM_EPOCH=NUM_EPOCH, saveDir=model.saveDir,
                   checkpoint=50, bestModelSave=True)
    model.get_performance_plt()


def test_predict(path):
    # testing dosen't use agumentation
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # data normalization
    ])

    test_set = ImgDataset(test.x, transforms=test_transforms)

    # upload to DataLoader
    BATCH_SIZE = 16
    NUM_WORKERS = 3  # depand on transforms step
    test_loader = DataLoader(test_set, BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 4 :0.62
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    model = HW3_Model(device, net, loss, optimizer)
    model.load_model(path)
    prediction = model.testing(test_loader)

    with open("{}/predict.csv".format(model.saveDir), 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))


if __name__ == "__main__":
    global img_inChannel, train, val, test
    img_inChannel = 3
    train = read_pickle('Data/train.pickle')
    val = read_pickle('Data/val.pickle')
    test = read_pickle('Data/test.pickle')

    for i, dataset in enumerate([train.x, val.x, test.x]):
        names = ['train', 'validation', 'test']
        print('Size of {} is {}'.format(names[i], len(dataset)))

    global device, net
    device = get_device()
    # net is use ntutiem_cyc server
    net = Net_Classifier_cyc(img_inChannel).to(device)

    # training_analysis(200, valiadating=True)
    pre_training('out/0823-0006/final_e200_0.0137.pickle', 100)
    # best_training()
    # test_predict('out/0820-1444/final_e200_0.0080.pickle')
