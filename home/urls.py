from django.urls import path

from home.views import *

urlpatterns =[
    path('test/', Test.as_view(), name='test'),
]