import os.path

import pandas as pd
import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
import matplotlib.image as mping
import h5py
import cv2
from scipy import spatial

from tensorflow.keras.layers import Flatten, Dense, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
import tensorflow as tf

from settings import PROJECT_ROOT

vgg16 = tensorflow.keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max',
                                            input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
basemodel.save(os.path.join(PROJECT_ROOT, 'basemodel.h5'))