import os
import glob
from io import BytesIO

import PIL.Image
from django.core.files.base import ContentFile
from django.http import HttpResponse
from PIL import Image
from django.shortcuts import render

# Create your views here.
from django.views import View

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

from FashionRecommender import settings
from .models import *

import sys

# load model (vgg, without output layer)
model = load_model(os.path.join(settings.PROJECT_ROOT, 'basemodel.h5'))

# set numpy to print the full thing
np.set_printoptions(threshold=sys.maxsize)

# define array shape, output of model and the dtype
ARRAY_SHAPE = (1, 4096)
ARRAY_DATA_TYPE = 'float32'


# these two functions are to generate feature vector and to calculate cosine similarity :)
def get_feature_vector(img):
    img1 = img.resize((224, 224))
    feature_vector = model.predict(np.asarray(img1).reshape((1, 224, 224, 3)))
    return feature_vector


def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


class Test(View):
    def get(self, request):
        img_path = os.path.join(settings.PROJECT_ROOT, '../dataset')
        for name in glob.glob(f'{img_path}/*'):
            print(name)
            for filename in glob.glob(f'{name}/*'):
                folder_name = name.split('\\')[-1]
                colour = folder_name.split('_')[0]
                type = folder_name.split('_')[-1]

                image_name = filename.split('\\')[-1]
                print(image_name)

                image = PIL.Image.open(filename)
                fv = get_feature_vector(image)

                buff = BytesIO()
                image.save(buff, format='jpeg')

                i = ImageDB(
                    colour=colour,
                    type=type,
                    image=ContentFile(buff.getvalue(), name=image_name),
                    gender='Female',
                )
                i.save()

                features = FeatureVector(
                    vector=fv.tostring(),
                    image_link=i,
                )
                features.save()

                # i is the current image item
                # features is the featurevector object(current item, stored as fv in mem)
                # j is each item iterated over, stored in FeatureVectors, including itself
                if FeatureVector.objects.all():
                    for j in FeatureVector.objects.all():
                        similarity = calculate_similarity(
                            fv,
                            np.frombuffer(j.vector, dtype=ARRAY_DATA_TYPE).reshape(ARRAY_SHAPE)
                        )

                        s = SimilarityMatrix(
                            column_item=i,
                            row_item=j.image_link,
                            value=similarity,
                        )
                        s.save()
                else:
                    s = SimilarityMatrix(
                        column_item=i,
                        row_item=i,
                        value=1,
                    )
                    s.save()
                # break

        return HttpResponse('okay')


class ImageUpload(View):
    def get(self, request):
        return render(request, 'img upload.html')

    def post(self, request):
        img = request.FILES.get('heh')
        print(img)
        i = ImageDB(
            colour='green',
            type='shorts',
            image=img,
            gender='Female',
        )
        i.save()
        fv = get_feature_vector(PIL.Image.open(i.image))
        f = FeatureVector(
            vector=fv.tostring(),
            image_link=i,
        )
        f.save()

        best_fits = list()
        for j in FeatureVector.objects.all():
            sim = calculate_similarity(
                fv,
                np.frombuffer(j.vector, dtype=ARRAY_DATA_TYPE).reshape(ARRAY_SHAPE)
            )
            s=SimilarityMatrix(
                column_item=i,
                row_item=j.image_link,
                value=sim,
            )
            s.save()
            if sim >= 0.5:
                best_fits.append(j)

        ctx = dict()
        ctx['recommend'] = best_fits
        print(best_fits)
        return render(request, 'img upload.html', context=ctx)