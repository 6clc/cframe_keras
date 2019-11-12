from cframe.models.classification import *


MODEL_DICT = dict(
    # efficient_net_b0=dict(
    #     name='efficient_net_b0',
    #     model=efficient_net_b0,
    #     config=dict()
    # ),
    resnet_v1=dict(
        name='resnet_v1',
        model=resnet_v1,
        config=dict(
            input_shape=(224, 224, 3),
            depth=8,
            num_classes=40
        )
    )
)


class ModelConfiger(object):
    @classmethod
    def get_model_names(cls):
        return MODEL_DICT.keys()
    @classmethod
    def get_model_config(cls, name):
        return MODEL_DICT[name]