import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import regularizers


def resnet50(config):
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    weights = config['weights']
    K.set_learning_phase(0)
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=input_shape)
    K.set_learning_phase(1)
    x = base_model.output
    # print('resnet backbone shape', x.shape)
    ap = GlobalAveragePooling2D(name='average_pool')(x)
    mp = GlobalMaxPooling2D(name='max_pool')(x)
    # print('ap ', ap.shape, 'mp ', mp.shape)
    x = K.concatenate([ap, mp], axis=-1)
    # print('cat', x.shape)
    # x = K.batch_flatten(x)
    # print('attention shape', x.shape)
    x = BatchNormalization()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
