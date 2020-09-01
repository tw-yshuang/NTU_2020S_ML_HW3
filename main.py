import torchvision.transforms as transforms
import torch.nn as nn
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from src.data_process import ImgDataset, DataPreProcess
from src.food_img_transform import train_transforms, test_transforms
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
    if valiadating:
        train_set = ImgDataset(
            train.x, train.y, train_transforms)  # 　<<<<<　train_transforms >>>>>
        val_set = ImgDataset(val.x, val.y, test_transforms)
        train_loader = DataLoader(train_set, BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS)
        try:
            model.training(train_loader, val_loader, NUM_EPOCH=NUM_EPOCH,
                           saveDir='./out', checkpoint=20, bestModelSave=True)
            model.get_performance_plt()
        except KeyboardInterrupt:
            model.get_performance_plt()
    else:
        train_set = ImgDataset(np.concatenate(
            (train.x, val.x)), np.concatenate((train.y, val.y)), train_transforms)
        train_loader = DataLoader(train_set, BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        model.training(train_loader, NUM_EPOCH=NUM_EPOCH,
                       saveDir='./out', checkpoint=20, bestModelSave=True)


def pre_training(path, NUM_EPOCH):
    # 　<<<<< train_transforms >>>>>
    train_set = ImgDataset(train.x, train.y, train_transforms)
    val_set = ImgDataset(val.x, val.y, test_transforms)

    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS)

    model.load_model(path, fullNet=(path.find('final_') != -1))
    try:
        model.training(train_loader, val_loader, NUM_EPOCH=NUM_EPOCH,
                       saveDir=model.saveDir, checkpoint=20, bestModelSave=True)
        model.get_performance_plt()
    except KeyboardInterrupt:
        model.get_performance_plt()


def test_predict(path):
    test_set = ImgDataset(test.x, transforms=test_transforms)
    test_loader = DataLoader(test_set, BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    model.load_model(path, fullNet=(path.find('final_') != -1))
    prediction = model.testing(test_loader)

    with open("{}/predict.csv".format(model.saveDir), 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))


def setting():
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

    global BATCH_SIZE, NUM_WORKERS
    # upload to DataLoader
    BATCH_SIZE = 32  # 　<<<<<　Batch size >>>>>
    NUM_WORKERS = 5  # depand on transforms step

    global model
    # training
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    model = HW3_Model(device, net, loss, optimizer)


if __name__ == "__main__":
    setting()

    training_analysis(150, valiadating=True)
    # pre_training('out/0828-1158/e100_0.0378.pickle', 250)
    # test_predict('out/0901-1308/e140_0.0258.pickle')

    # a = torch.load('out/0825-1750/best_e006_0.0570.pickle')
    # print(a)
