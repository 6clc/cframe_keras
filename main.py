from cframe.dataloader import DataConfiger
from cframe.dataloader import ClassificationDataloaderManager
from cframe.models import ModelConfiger
from cframe.models import ModelManager
from cframe.learner import BasicLearner
from keras.datasets import cifar10
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    from keras import backend as K

    K.tensorflow_backend._get_available_gpus()
    data_config = DataConfiger.get_data_config('garbage')
    dl_manager = ClassificationDataloaderManager(data_config)

    model_config = ModelConfiger.get_model_config('resnet50')
    model_manager = ModelManager(model_config)

    learner = BasicLearner(model_manager, dl_manager,
                           optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    learner.train(3)




