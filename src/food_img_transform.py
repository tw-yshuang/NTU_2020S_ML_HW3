import cv2
import skimage
import random
import numpy as np
from torchvision import transforms
try:
    from src.Model.find_file_name import get_filenames
except ModuleNotFoundError:
    from Model.find_file_name import get_filenames

train_transforms_arg = transforms.Compose([
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=(2)),
        transforms.ColorJitter(contrast=(0.5, 3)),
        transforms.ColorJitter(saturation=(0.5, 5)),
        transforms.ColorJitter(hue=(-0.20, 0.05))
    ]),

    transforms.ColorJitter(brightness=(0.75, 1.25), contrast=(
        0.75, 1.25), saturation=(0.15, 1.15), hue=(-0.10, 0.1)),

    transforms.Pad(padding=(32, 32), padding_mode='symmetric'),
    transforms.RandomRotation(45, expand=False),
    transforms.Lambda(lambda x: cv2_transforms(x)),
    transforms.ToPILImage(),
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


def cv2_transforms(img, size=128, isShow=False):
    img = np.asarray(img)
    img_center = img.shape[0] // 2
    half_size = size // 2
    crop_img = img[img_center-half_size: img_center +
                   half_size, img_center-half_size: img_center+half_size]
    output_img = random.choice(
        [get_pepper_salt_noised(crop_img, 0.0025), crop_img])

    isShow:
        cv2.imshow('img', img)
        cv2.imshow('crop', crop_img)
        cv2.imshow('output', output_img)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return output_img


def get_pepper_salt_noised(img, amount=0.05, isShow=False):
    # img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    img = img/255.0  # floating point image
    img_noised = skimage.util.random_noise(img, 's&p', amount=amount)
    img_noised = np.uint8(img_noised*256)
    # cvImg_noised = cv2.cvtColor(img_noised, cv2.COLOR_BGR2RGB)
    if isShow:
        cv2.imshow("Pepper_salt_noise: " + str(amount), img_noised)
        cv2.waitKey(0)
    return img_noised


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filenames = get_filenames('Data/food-11/training', '.jpg')
    for filename in filenames:
        img = cv2.imread(filename)
        # img = get_pepper_salt_noised(img, 0.05, True)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process_img = train_transforms(img)

        # process_img = cv2.imread(process_img)
        # cv2.imshow('raw', img)
        # cv2.imshow('process_img', process_img)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(process_img)
        plt.show(block=False)
        plt.pause(0.05)
        plt.cla()
        # plt.waitforbuttonpress(0)
        # plt.close()
        # if 0xFF == ord('q'):
        #     print('q')
        #     plt.close('all')
