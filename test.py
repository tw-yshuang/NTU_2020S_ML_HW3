# import time
# import numpy as np
# a = np.array([1, 0, 0, 1, 1])
# b = np.array([1, 1, 0, 1, 1])

# print('a', a)
# print('b', b)
# print(np.sum(a == b))


# print(time.localtime())

# import os

# try:
#     from src.Model.find_file_name import get_filenames
# except ModuleNotFoundError:
#     from Model.find_file_name import get_filenames

# filenames = get_filenames('out/', '*e010*.pkl')
# print(filenames)

# path = "out/"
# files = os.listdir(path)
# for i, f in enumerate(files):
#     if f.find("e050") >= 0:
#         print(i)
#         os.remove(path+f)
# os.remove('out/*e050*.pkl')
# print('delete')

import pickle
import torch

from src.net import Net_Classifier_cyc


def read_pickle(path):
    with open(path, 'rb') as target:
        return pickle.load(target)


a = read_pickle('out/0818-1504/e200_0.0041.pickle')
# a = torch.load('out/0816-1741/best_e020_0.0084.pickle')
net = Net_Classifier_cyc(3)
a.net = net.load_state_dict(a.net)
print(a)

# dict_1 = {'a': [2, 5, 4, 7, 10], 'b': [45, 25, 6, 3, 5]}
# print(dict_1)
