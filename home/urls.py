from django.urls import path

from home.views import *

urlpatterns =[
    path('test/', Test.as_view(), name='test'),
    path('image/', ImageUpload.as_view(), name='img-upload'),
    path('gg/', XYZ.as_view(), name='xyz'),
]