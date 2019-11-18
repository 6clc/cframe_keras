import numpy as np
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers


def resnet50(config):
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    RESNET50_POOLING_AVERAGE = 'avg'
    DENSE_LAYER_ACTIVATION = 'softmax'

    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights=config['weights']))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(num_classes, activation=DENSE_LAYER_ACTIVATION))

    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False
    return model

# def resnet50(config):
#     input_shape = config['input_shape']
#     num_classes = config['num_classes']
#     weights = config['weights']
#     K.set_learning_phase(0)
#     base_model = ResNet50(weights=weights, include_top=False,
#                           input_shape=input_shape)
#     K.set_learning_phase(1)
#     x = base_model.output
#     # print('resnet backbone shape', x.shape)
#     ap = GlobalAveragePooling2D(name='average_pool')(x)
#     mp = GlobalMaxPooling2D(name='max_pool')(x)
#     # print('ap ', ap.shape, 'mp ', mp.shape)
#     x = layers.concatenate([ap, mp], axis=-1)
#     # print('cat', x.shape)
#     # x = K.batch_flatten(x)
#     # print('attention shape', x.shape)
#     x = BatchNormalization()(x)
#     x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
#     x = BatchNormalization()(x)
#     x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
#     x = BatchNormalization(name='bn_fc_01')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     return model
