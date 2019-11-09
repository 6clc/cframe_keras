from cframe.models.classification import *


MODEL_DICT = dict(
    simplenet=dict(
        name='simplenet',
        model=SimpleNet,
        config=dict(input_shape=(224, 224, 3))
    )
)


class ModelConfiger(object):
    @classmethod
    def get_model_names(cls):
        return MODEL_DICT.keys()
    @classmethod
    def get_model_config(cls, name):
        return MODEL_DICT[name]