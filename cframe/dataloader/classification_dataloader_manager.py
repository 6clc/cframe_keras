from cframe.dataloader.dataloader import *


class ClassificationDataloaderManager(object):
    def __init__(self, config):
        self.config = config

    def get_train_dl(self):
        return ClassificationDataloader(config=self.config,
                                       mode='train',
                                       shuffle=True
                                       )

    def get_valid_dl(self):
        return ClassificationDataloader(config=self.config,
                                       mode='valid',
                                       shuffle=False)

    def get_test_dl(self):
        return ClassificationDataloader(config=self.config,
                                       mode='test',
                                       shuffle=False)

