from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import pandas as pd
import os
import cv2


class ClassificationGenerator(Sequence):
    def __init__(self, config, mode, shuffle, random_state=2019):
        self.config = config
        self.mode = mode
        self.root_dir = self.config['root_dir']
        self.batch_size = self.config['batch_size']
        self.shuffle = shuffle
        self.random_state = random_state
        self.dim = self.config['dim']
        self.n_channels = self.config['n_channels']
        self.n_classes = self.config['n_classes']
        self.root_dir = self.config['root_dir']
        self.dataset_dir = self.config['dataset_dir']
        self.csv_dir = self.config['csv_dir']
        self.data_name = self.config['data_name']
        self.mode = mode
        self.on_epoch_end()
        self.df = pd.read_csv(
            os.path.join(self.root_dir,
                         self.csv_dir,
                         self.data_name,
                         '{}.csv'.format(mode))
        )
        self.list_IDs = [i for i in range(len(self.df))]

        self._init()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)
        y = self.__generate_y(list_IDs_batch)
        return X, y

    def _init(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def  __generate_X(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.df['img'][ID]
            img_path = os.path.join(self.root_dir, img_name)
            img = self.__load_rgb(img_path)
            X[i, ] = img

        return X

    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, 1), dtype=int)
        for i, ID in enumerate(list_IDs_batch):
            label = self.df['label'][ID]
            y[i, 0] = label
        return y

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.dim, cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.
        return img


if __name__ == '__main__':
    config = dict(root_dir='/Volumes/data2/Data',
                  dataset_dir='DataSets',
                  data_name='leaf-classification',
                  csv_dir='CSVs',
                  n_classes=20,
                  batch_size=4,
                  dim=(224, 224),
                  n_channels=3,
                  )
    dg = ClassificationGenerator(config, mode='train', shuffle=True)
    for X, y in dg:
        print(X.shape, y)
