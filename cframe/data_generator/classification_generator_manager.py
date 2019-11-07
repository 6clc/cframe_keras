from cframe.data_generator.data_generator import *


class ClassificationGeneratorManager(object):
    def __init__(self, config):
        self.config = config

    def get_train_dl(self):
        return ClassificationGenerator(config=self.config,
                                       mode='train',
                                       shuffle=True
                                       )

    def get_valid_dl(self):
        return ClassificationGenerator(config=self.config,
                                       mode='valid',
                                       shuffle=False)

    def get_test_dl(self):
        return ClassificationGenerator(config=self.config,
                                       mode='test',
                                       shuffle=False)


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
    dl_manager = ClassificationGeneratorManager(config)
    train_dl = dl_manager.get_train_dl()
    print(len(train_dl))