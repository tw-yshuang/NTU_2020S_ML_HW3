# import time
# import numpy as np
# a = np.array([1, 0, 0, 1, 1])
# b = np.array([1, 1, 0, 1, 1])

# print('a', a)
# print('b', b)
# print(np.sum(a == b))


# print(time.localtime())

import os

try:
    from src.Model.find_file_name import get_filenames
except ModuleNotFoundError:
    from Model.find_file_name import get_filenames

filenames = get_filenames('out/', '*e010*.pkl')
print(filenames)

# path = "out/"
# files = os.listdir(path)
# for i, f in enumerate(files):
#     if f.find("e050") >= 0:
#         print(i)
#         os.remove(path+f)
# os.remove('out/*e050*.pkl')
# print('delete')
