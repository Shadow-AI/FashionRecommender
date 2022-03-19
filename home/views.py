import json
import os
import glob
import time
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
    # img2 = tf.image.resize(img, (224, 224))
    feature_vector = model.predict(np.asarray(img1).reshape((1, 224, 224, 3)))
    return feature_vector


def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


class Test(View):
    def get(self, request):
        print('hhe')
        start = time.time()

        dataset_path = os.path.join(settings.PROJECT_ROOT, '../dataset')
        image_path = os.path.join(settings.PROJECT_ROOT, '../dataset/images')
        styles_path = os.path.join(settings.PROJECT_ROOT, '../dataset/styles')

        for iter_no, name in enumerate(glob.glob(f'{styles_path}/*')):
            image_name = name.split('\\')[-1].split('.')[0]
            # todo you were working here; need to load images to dataset. dont need image_path ig
            with open(name, 'r') as fp:

                try:
                    item_json = json.load(fp)
                except UnicodeDecodeError:
                    continue

                buy_link = f"myntra.com/{item_json['data']['landingPageUrl']}"

                if item_json['data']['styleImages'].get('front'):
                    pic_front = item_json['data']['styleImages']['front']['imageURL']
                else:
                    pic_front = item_json['data']['styleImages']['default']['imageURL']
                if item_json['data']['styleImages'].get('back'):
                    pic_back = item_json['data']['styleImages']['back'].get('imageURL')
                else:
                    pic_back = None
                if item_json['data']['styleImages'].get('left'):
                    pic_left = item_json['data']['styleImages']['left'].get('imageURL')
                else:
                    pic_left = None
                if item_json['data']['styleImages'].get('right'):
                    pic_right = item_json['data']['styleImages']['right'].get('imageURL')
                else:
                    pic_right = None

                item_name = item_json['data']['productDisplayName']
                item_colour = item_json['data']['baseColour']
                item_gender = item_json['data']['gender']
                item_agegrp = item_json['data']['ageGroup']
                item_brand = item_json['data']['brandName']
                item_usage = item_json['data']['usage']
                item_season = item_json['data']['season']
                # todo do you want to put display categories too?

                item_main = item_json['data']['masterCategory']['typeName']
                item_sub = item_json['data']['subCategory']['typeName']
                item_article = item_json['data']['articleType']['typeName']

                # name is actually the full path to the json
                image = PIL.Image.open(f'{image_path}/{image_name}.jpg')
                fv = get_feature_vector(image)

                # buff = BytesIO()
                # image.save(buff, format='jpeg')

                i = ImageObject(
                    name=item_name,
                    colour=item_colour,
                    image_link_front=pic_front,
                    image_link_back=pic_back,
                    image_link_left=pic_left,
                    image_link_right=pic_right,
                    image_buy_link=buy_link,
                    is_custom=False,
                    gender=item_gender,
                    age_group=item_agegrp,
                    brand=item_brand,
                    season=item_season,
                    usage=item_usage,
                    main_category=item_main,
                    sub_category=item_sub,
                    article_category=item_article,
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
                # if FeatureVector.objects.all():
                #     for j in FeatureVector.objects.all():
                #         similarity = calculate_similarity(
                #             fv,
                #             np.frombuffer(j.vector, dtype=ARRAY_DATA_TYPE).reshape(ARRAY_SHAPE)
                #         )
                #
                #         s = SimilarityMatrix(
                #             column_item=i,
                #             row_item=j.image_link,
                #             value=similarity,
                #         )
                #         s.save()
                # else:
                #     s = SimilarityMatrix(
                #         column_item=i,
                #         row_item=i,
                #         value=1,
                #     )
                #     s.save()

            os.remove(f'{image_path}/{image_name}.jpg')
            os.remove(name)

            # if (iter_no + 1) % 500 == 0:
            #     break

        return HttpResponse(time.time() - start)


class ImageUpload(View):
    def get(self, request):
        ctx = {
            'USAGE': USAGE,
            'SEASONS': SEASONS,
            'AGE_GROUP': AGE_GROUP,
            'GENDER': GENDER,
        }
        return render(request, 'img upload.html', context=ctx)

    def post(self, request):
        # todo this needs to be revamped
        img = request.FILES.get('user-img')

        print(img)
        start = time.time()
        # i = ImageObject(
        #     colour=request.POST.get('user-colour'),
        #     image=img,
        #     gender=request.POST.get('user-gender'),
        #     is_custom=True,
        #     age_group=request.POST.get('user-agegrp'),
        #     brand=request.POST.get('user-brand'),
        #     season=request.POST.get('user-season'),
        #     usage=request.POST.get('user-usage'),
        # )
        # i.save()
        fv = get_feature_vector(PIL.Image.open(img))
        # f = FeatureVector(
        #     vector=fv.tostring(),
        #     image_link=i,
        # )
        # f.save()

        best_fits = list()
        for j in FeatureVector.objects.all():
            sim = calculate_similarity(
                fv,
                np.frombuffer(j.vector, dtype=ARRAY_DATA_TYPE).reshape(ARRAY_SHAPE)
            )
            # s = SimilarityMatrix(
            #     column_item=i,
            #     row_item=j.image_link,
            #     value=sim,
            # )
            # s.save()
            if sim >= 0.75:
                best_fits.append(j)

        ctx = {
            'USAGE': USAGE,
            'SEASONS': SEASONS,
            'AGE_GROUP': AGE_GROUP,
            'GENDER': GENDER,
            'recommend': best_fits,
        }
        # print(best_fits)
        print(time.time() - start)
        return render(request, 'img upload.html', context=ctx)

class XYZ(View):
    def get(self, request):
        p = int(request.GET.get('i'))
        if p == 1:
            return render(request, 'base.html')
        elif p == 2:
            return render(request, 'cart.html')
        elif p == 3:
            return render(request, 'checkout.html')
        elif p == 4:
            return render(request, 'index.html')
        elif p == 5:
            return render(request, 'product-details.html')
        elif p == 6:
            return render(request, 'shop.html')

        # todo checkout X, cart -> wishlist,
        # todo designer bags/sale -> kids wagera ad

# actual web code begins here:

class Index(View):
    def get(self, request):
        ctx = dict()
        return render(request, 'index.html', context=ctx)

class DisplayRecommendation(View):
    def get(self, request):
        ctx = dict()

        return render(request, 'shop.html', context=ctx)

    def post(self, request):
        start = time.time()
        img = request.FILES.get('rec-img')
        fv = get_feature_vector(PIL.Image.open(img))
        best_fits = list()
        usage = set()
        season = set()
        gender = set()
        main_category = set()
        colour = set()
        best_fits = dict()
        for i in FeatureVector.objects.all():
            sim = calculate_similarity(
                fv,
                np.frombuffer(i.vector, dtype=ARRAY_DATA_TYPE).reshape(ARRAY_SHAPE)
            )
            if sim >= 0.75:
                colour.add(i.image_link.colour)
                main_category.add(i.image_link.main_category)
                gender.add(i.image_link.gender)
                season.add(i.image_link.season)
                usage.add(i.image_link.usage)
                t = list()
                if i.image_link.image_link_front:
                    t.append(i.image_link.image_link_front)
                if i.image_link.image_link_back:
                    t.append(i.image_link.image_link_back)
                if i.image_link.image_link_left:
                    t.append(i.image_link.image_link_left)
                if i.image_link.image_link_right:
                    t.append(i.image_link.image_link_right)
                best_fits[i] = t



# todo send few top results only not all
# best fits is a dict with key as the actual item(fv) and the value as the images associated

        ctx = {
            'recommend': best_fits,
            'total time': time.time() - start,
            'colour': colour,
            'main_category': main_category,
            'gender': gender,
            'season': season,
            'usage': usage,
        }

        return render(request, 'shop.html', context=ctx)

