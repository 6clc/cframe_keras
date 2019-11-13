from tensorflow.keras.preprocessing.image import ImageDataGenerator
bs = 64
data_root_dir = '/home/he/Data'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    data_root_dir + '/DataSets/garbage_classify/imgs',
    target_size=(224, 224),
    batch_size=bs,
    color_mode='rgb',
    class_mode='categorical'
)

# from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import ResNet50
model = ResNet50(weights=None,
                     input_shape=(224, 224, 3),
                     classes=40)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=14802/bs, epochs=5)

import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
