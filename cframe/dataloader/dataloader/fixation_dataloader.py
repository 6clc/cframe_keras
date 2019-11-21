from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import pandas as pd
import os
import cv2
from cframe.dataloader.tools import standard_transform as std_trans


class FixationDataloader(Sequence):
    def __init__(self, config, mode, shuffle, random_state=2019):
        self.config = config
        self.mode = mode
        self.root_dir = self.config['root_dir']
        self.batch_size = self.config['batch_size']
        self.shuffle = shuffle
        self.random_state = random_state

        self.resize = self.config['resize']
        self.in_channels = self.config['in_channels']

        self.root_dir = self.config['root_dir']
        self.dataset_dir = self.config['dataset_dir']
        self.csv_dir = self.config['csv_dir']
        self.data_name = self.config['data_name']
        self.mode = mode

        self.df = pd.read_csv(
            os.path.join(self.root_dir,
                         self.csv_dir,
                         self.data_name,
                         '{}.csv'.format(mode))
        )
        self.list_IDs = [i for i in range(len(self.df))]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        images = self.__generate_images(list_IDs_batch)
        maps = self.__generate_maps(list_IDs_batch)
        fixs = self.__generate_fixmaps(list_IDs_batch)
        return [images], [maps, maps, fixs]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
        # print('shuffle again')

    def __generate_images(self, list_IDs_batch):
        imgs = np.empty((self.batch_size, *self.resize, self.in_channels))
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.df['img'][ID]
            img_path = os.path.join(self.root_dir, img_name)
            ori_img = cv2.imread(img_path)
            padded_img = std_trans.padding(ori_img, *self.resize, 3)
            imgs[i] = padded_img

        imgs[:, :, :, 0] -= 103.939
        imgs[:, :, :, 1] -= 116.779
        imgs[:, :, :, 2] -= 123.68
        imgs = imgs.transpose((0, 3, 1, 2))

        return imgs

    def __generate_maps(self, list_IDs_batch):
        imgs = np.zeros((self.batch_size, 1, *self.resize))
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.df['saliency'][ID]
            img_path = os.path.join(self.root_dir, img_name)
            ori_map = cv2.imread(img_path, 0)
            padded_map = std_trans.padding(ori_map, *self.resize, 1)
            imgs[i, 0] = padded_map.astype(np.float32)
            imgs[i, 0] /= 255.
        return imgs

    def __generate_fixmaps(self, list_IDs_batch):
        imgs = np.zeros((self.batch_size, 1, *self.resize))
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.df['fixation'][ID]
            img_path = os.path.join(self.root_dir, img_name)
            fix_map = cv2.imread(img_path, 0)
            fix_map[fix_map == 255] = 1
            fix_map = std_trans.padding_fixation(fix_map, *self.resize)
            imgs[i, 0] = fix_map
        return imgs


    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize, cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.
        return img


if __name__ == '__main__':
    config = dict(root_dir='/Volumes/data2/Data',
                  dataset_dir='DataSets',
                  data_name='leaf-classification',
                  csv_dir='CSVs',
                  n_classes=20,
                  batch_size=4,
                  resize=(224, 224),
                  in_channels=3,
                  )
    dg = ClassificationDataloader(config, mode='train', shuffle=True)
    for X, y in dg:
        print(X.shape, y)
