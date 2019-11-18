from cframe.dataloader import DataConfiger
from cframe.dataloader import ClassificationDataloaderManager
from cframe.learner import BasicLearner
from cframe.models import ModelConfiger
from cframe.models import ModelManager
import os

root_dir = '/content/drive/My Drive'

data_config = DataConfiger.get_data_config('garbage')
data_config['root_dir'] = os.path.join(root_dir, 'Data')
data_config['batch_size'] = 32
dl_manager = ClassificationDataloaderManager(data_config)

model_config = ModelConfiger.get_model_config('resnet50')
model_manager = ModelManager(model_config)

learner = BasicLearner(model_manager, dl_manager,
                       loss='sparse_categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])


learner.train(5)
