from django.core.files.storage import FileSystemStorage
from django.db import models
from django.utils.html import escape

# Create your models here.

fs = FileSystemStorage()

CLOTH_TYPE = (
    ('dress', 'Dress'),
    ('shirt', 'Shirt'),
    ('pants', 'Pants'),
    ('shorts', 'Shorts'),
    ('shoes', 'Shoes'),
)

GENDER = (
    ('Male', 'Male'),
    ('Female', 'Female'),
)


class ImageDB(models.Model):
    name = models.CharField(max_length=200, null=True, blank=True)
    colour = models.CharField(max_length=50)
    type = models.CharField(max_length=50, choices=CLOTH_TYPE)
    image = models.ImageField()
    gender = models.CharField(max_length=10, choices=GENDER, null=True, blank=True)
    # todo think about use uploading image, for colour, type and gender
    # 1. accept from user
    # 2. profit

    def save(self, *args, **kwargs):
        # comment this out if need name to be something diff, idk why tho (idk why name field there)
        self.name = self.image.name
        super(ImageDB, self).save(*args, **kwargs)

    def delete(self, using=None, keep_parents=False):
        fs.delete(self.image.name)
        super().delete()

    def __str__(self):
        return f'{self.type} | {self.colour} | {self.name}'


class FeatureVector(models.Model):
    # vector is stored in bytes object via pickle. use loads to load it as array
    vector = models.BinaryField()
    image_link = models.ForeignKey(ImageDB, on_delete=models.CASCADE)
    array_datatype = models.CharField(max_length=10, default='float32')

    def __str__(self):
        return f'{self.image_link}'


class SimilarityMatrix(models.Model):
    column_item = models.ForeignKey(ImageDB, on_delete=models.CASCADE, related_name='column_item')
    row_item = models.ForeignKey(ImageDB, on_delete=models.CASCADE, related_name='row_item')
    value = models.FloatField()

    def __str__(self):
        col_name = self.column_item
        row_name = self.row_item
        return f'{self.column_item} X {self.row_item}'
