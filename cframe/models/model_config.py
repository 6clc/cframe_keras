from cframe.models.classification import *


MODEL_DICT = dict(

)


class ModelConfiger(object):
    @classmethod
    def get_model_names(cls):
        return MODEL_DICT.keys()
    @classmethod
    def get_model_config(cls, name):
        return MODEL_DICT[name]