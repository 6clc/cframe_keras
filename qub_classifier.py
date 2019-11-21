import numpy as np
from skimage.io import imread
from skimage.transform import resize
# from keras.applications.imagenet_utils import decode_predictions
# from classification_models.keras import
from cframe.models import QubClassifiers

ResNet18, preprocess_input = QubClassifiers.get('resnet18')
