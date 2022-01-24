from django.contrib import admin

# Register your models here.

from home.models import *
admin.site.register(ImageDB)
admin.register(FeatureVector)

