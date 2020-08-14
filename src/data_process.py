# import torchvision.transforms as transforms
import torch
import cv2
import pickle
import numpy as np
from src.Model.find_file_name import get_filenames
from torch.utils.data import Dataset


class DataPreProcess(object):
    def __init__(self, path, imgShape, imgChannel, isTest=False, savePath=None):
        self.imgChannel = imgChannel
        self.path = path
        self.savePath = savePath
        self.filenames = get_filenames(self.path, 'jpg')

        if imgShape is not tuple:
            imgShape = (imgShape, imgShape)
        self.imgShape = imgShape

        if isTest is False:
            self.x, self.y = self.get_dataset(isTest)
        else:
            self.x = self.get_dataset(isTest)

        if self.savePath is not None:
            with open(self.savePath, 'wb') as target:
                pickle.dump(self, target)

    def get_dataset(self, isTest=False):
        x = np.zeros([len(self.filenames), self.imgShape[0],
                      self.imgShape[1], self.imgChannel], dtype='uint8')
        y = np.zeros([len(self.filenames)], dtype='uint8')

        if isTest is False:
            for i, filename in enumerate(self.filenames):
                img = cv2.imread(filename)
                if img.shape[2] != self.imgChannel:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x[i] = cv2.resize(img, self.imgShape)
                y[i] = str.split(filename, '/')[-1].split('_')[-2]
            return x, y

        else:
            for i, filename in enumerate(self.filenames):
                img = cv2.imread(filename)
                x[i] = cv2.resize(img, self.imgShape)
            return x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        self.x = x
        if y is not None:
            # self.y = y
            # y = self.one_hot_encode()
            self.y = torch.LongTensor(y)
        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transforms is not None:
            X = self.transforms(X)

        if self.y is not None:
            Y = self.y[index]
            return X, Y

        return X

    def one_hot_encode(self):
        num_class = max(self.y) + 1
        y_array = np.zeros([len(self.y), num_class])
        for i, y in enumerate(self.y):
            y_array[i][y] = 1

        return y_array


if __name__ == "__main__":
    img_inChannel = 3
    train = DataPreProcess('Data/food-11/training', 128,
                           img_inChannel, savePath='Data/train.pickle')
    val = DataPreProcess('Data/food-11/validation', 128,
                         img_inChannel, savePath='Data/val.pickle')
    test = DataPreProcess('Data/food-11/testing', 128,
                          img_inChannel, isTest=True,  savePath='Data/test.pickle')
