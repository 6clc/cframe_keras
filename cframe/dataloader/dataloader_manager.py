from cframe.dataloader.dataloader import *

DL_DICT = dict(
    fixation=FixationDataloader,
    classification=ClassificationDataloader
)


class DataloaderManager(object):
    def __init__(self, config, task):
        self.config = config
        assert task in DL_DICT.keys()
        self.task = task

    def get_train_dl(self):
        return DL_DICT[self.task](config=self.config,
                                  mode='train',
                                  shuffle=True
                                  )

    def get_valid_dl(self):
        return DL_DICT[self.task](config=self.config,
                                  mode='valid',
                                  shuffle=False)

    def get_test_dl(self):
        return DL_DICT[self.task](config=self.config,
                                  mode='test',
                                  shuffle=False)


if __name__ == '__main__':
    from cframe.dataloader.data_configer import DataConfiger
    config = DataConfiger.get_data_config('SALICON')
    dl_manager = DataloaderManager(config, task='fixation')
    train_dl = dl_manager.get_train_dl()
    X, y = next(iter(train_dl))
    print(y[2].shape, np.unique(y[2]))
    print(y[1].shape, np.unique(y[1]))
    from matplotlib import pyplot as plt
    plt.hist(X[0].reshape(-1))
    plt.show()