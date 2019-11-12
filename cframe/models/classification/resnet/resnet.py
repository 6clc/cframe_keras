import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import backend as K


K.set_learning_phase(0)
Inp = Input((224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=(image_size, image_size, 3), )
x = base_model(Inp)
x = base_model.output
x = GlobalAveragePooling2D(name='average_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = BatchNormalization(name='bn_fc_01')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)