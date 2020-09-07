import torch
import cv2
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

try:
    from src.Model.find_file_name import get_filenames
except ModuleNotFoundError:
    from Model.find_file_name import get_filenames


class DataPreProcess(object):
    def __init__(self, path, imgShape, imgChannel, isTest=False, dataBalance=False, savePath=None):
        self.imgChannel = imgChannel
        self.path = path
        self.savePath = savePath
        self.dataBalance = dataBalance
        self.filenames = sorted(get_filenames(self.path, '.jpg'))

        if imgShape is not tuple:
            imgShape = (imgShape, imgShape)
        self.imgShape = imgShape

        if isTest is False:
            if dataBalance is True:
                self.get_dataBalance()
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

    # calculate how many filename in each classifier
    def calculate_classifier(self):
        classifier_dict = {}
        for filename in self.filenames:
            num_classifier = str.split(filename, '/')[-1].split('_')[-2]
            classifier_dict[num_classifier] = classifier_dict.get(
                num_classifier, 0) + 1
        # print(classifier_dict)
        self.classifier_dict = classifier_dict

    def get_dataBalance(self):
        self.calculate_classifier()
        max_num_classifier = max(self.classifier_dict.values())
        copy_filenames_dict = {}
        for num_classifier in self.classifier_dict.keys():
            # get classifier type and calculate the gap between max_num_classifier and num_type_classifier
            copy_filenames_dict.update(
                {num_classifier: [(max_num_classifier - self.classifier_dict[num_classifier])]})

        for filename in self.filenames:
            num_classifier = str.split(filename, '/')[-1].split('_')[-2]
            if self.classifier_dict[num_classifier] < max_num_classifier:
                times_data_gap = max_num_classifier / \
                    self.classifier_dict[num_classifier]
                for i in range(int(times_data_gap)):
                    copy_filenames_dict[num_classifier].append(filename)

        # random select filename copy into filenames
        for k, v in copy_filenames_dict.items():
            select_times = v.pop(0)
            self.filenames.extend(random.sample(v, select_times))
        # self.calculate_classifier()


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        self.x = x
        self.y = y
        if self.y is not None:
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
    # train = DataPreProcess('Data/food-11/training', 128,
    #                        img_inChannel, dataBalance=True, savePath='Data/train_balance.pickle')
    val = DataPreProcess('Data/food-11/validation', 256,
                         img_inChannel, savePath='Data/val.pickle')
    # test = DataPreProcess('Data/food-11/testing', 128,
    #                       img_inChannel, isTest=True,  savePath='Data/test.pickle')
