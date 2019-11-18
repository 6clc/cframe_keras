from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from cframe.models import ModelConfiger
from cframe.models import ModelManager
import os

root_dir = '/content/drive/My Drive'
BATCH_SIZE_TRAINING = 100
num_epochs = 5
num_images = 14802
STEPS_PER_EPOCH_TRAINING = num_images // BATCH_SIZE_TRAINING

model_config = ModelConfiger.get_model_config('resnet50')
model_manager = ModelManager(model_config)
model = model_manager.get_model()

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_directory(
        os.path.join(root_dir, 'Data/DataSets/garbage_classify/imgs'),
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = num_epochs,
)

