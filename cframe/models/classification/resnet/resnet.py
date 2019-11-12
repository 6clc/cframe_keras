import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras import regularizers


def resnet50(config):
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    K.set_learning_phase(0)
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=input_shape )
    K.set_learning_phase(1)

    x = base_model.output
    # print(x.shape, type(x))
    x = GlobalAveragePooling2D(name='average_pool')(x)
    # print(x.shape)
    # x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
