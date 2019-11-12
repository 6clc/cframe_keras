class ModelManager(object):
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = None
        self._init()

    def _init(self):
        self.model = self.model_config['model'](**self.model_config['config'])

    def get_model(self):
        return self.model
