from cframe.models import ModelManager
import os
import tensorflow  as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

class BasicLearner(object):
    def __init__(self, model_manager: ModelManager,
                 dl_manager,
                 loss, optimizer, metrics, callbacks=None):
        if callbacks is None:
            callbacks = []
        self.model_manager = model_manager
        self.dl_manager = dl_manager
        self.callbacks = callbacks

        self.model = self.model_manager.get_model()
        self.train_dl = self.dl_manager.get_train_dl()
        self.valid_dl = self.dl_manager.get_valid_dl()
        self.test_dl = self.dl_manager.get_test_dl()

        self.summary_writer_dir = None

        self._init()

        checkpoint = ModelCheckpoint(filepath=os.path.join(self.summary_writer_dir, 'best.hdf5'),
                                     monitor='val_acc', save_best_only=True, verbose=1, mode='max')
        self.callbacks.append(checkpoint)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)



    def _init(self):
        root_dir = '/'.join(self.dl_manager.config['root_dir'].split('/')[:-1]) + '/SummaryWriter'
        data_name = self.dl_manager.config['data_name']
        model_name = self.model_manager.model_config['name']
        self.summary_writer_dir = os.path.join(root_dir, '_'.join([data_name, model_name]))
        if not os.path.exists(self.summary_writer_dir):
            os.makedirs(self.summary_writer_dir)

    def train(self, num_epoches):
        history = self.model.fit_generator(
            self.train_dl,
            validation_data=self.valid_dl,
            epochs=num_epoches,
            callbacks=self.callbacks
        )
        return history
        # -

