from django.contrib import admin

# Register your models here.
from django.utils.html import format_html

from home.models import *
@admin.register(ImageDB)
class ImageDBAdmin(admin.ModelAdmin):
    list_filter = ['colour', 'type']
    list_display = ['name', 'gender']
    fields = ('name', 'colour', 'type', 'image', 'gender', 'image_tag')
    readonly_fields = ('image_tag',)

    def image_tag(self, obj):
        return format_html('<img src="{}" />'.format(obj.image.url))

    image_tag.short_description = 'Image'


    class Meta:
        model = ImageDB




admin.site.register(FeatureVector)
admin.site.register(SimilarityMatrix)