from cframe.models import ModelManager
from cframe.data_generator import ClassificationGeneratorManager
import os


class BasicLearner(object):
    def __init__(self, model_manager: ModelManager,
                 dl_manager: ClassificationGeneratorManager,
                 loss, optimizer, metrics):
        self.model_manager = model_manager
        self.dl_manager = dl_manager

        self.model = self.model_manager.get_model()
        self.train_dl = self.dl_manager.get_train_dl()
        self.valid_dl = self.dl_manager.get_valid_dl()
        self.test_dl = self.dl_manager.get_test_dl()

        self.summary_writer_dir = None
        self.writer = None

        self.running_score = dict()
        self.best_para = None

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

        self._init()

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
            use_multiprocessing=False,
            validation_data=self.valid_dl,
            workers=1,
            epochs=num_epoches
        )
        return history

