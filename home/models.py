from django.core.files.storage import FileSystemStorage
from django.db import models

# Create your models here.

fs = FileSystemStorage()

CLOTH_TYPE = (
    ('Dress', 'Dress'),
    ('Shirt', 'Shirt'),
    ('Pants', 'Pants'),
    ('Shorts', 'Shorts'),
    ('Shoes', 'Shoes'),
)

GENDER = (
    ('Male', 'Male'),
    ('Female', 'Female'),
)


class ImageDB(models.Model):
    name = models.CharField(max_length=200, null=True, blank=True)
    colour = models.CharField(max_length=50)
    type = models.CharField(max_length=50, choices=CLOTH_TYPE)
    image = models.ImageField(upload_to='media')
    gender = models.CharField(max_length=10, choices=GENDER, null=True, blank=True)


    def save(self, *args, **kwargs):
        # comment this out if need name to be something diff, idk why tho (idk why name field there)
        self.name = self.image.name
        super(ImageDB, self).save(*args, **kwargs)

    def delete(self, using=None, keep_parents=False):
        print('1')
        fs.delete(self.image.name)
        print('2')
        super().delete()

    def __str__(self):
        return f'{self.type} | {self.colour} | {self.name}'


class FeatureVector(models.Model):
    vector = models.TextField()
    image_link = models.ForeignKey(ImageDB, on_delete=models.CASCADE)


class SimilarityMatrix(models.Model):
    column_item = models.ForeignKey(ImageDB, on_delete=models.CASCADE, related_name='column_item')
    row_item = models.ForeignKey(ImageDB, on_delete=models.CASCADE, related_name='row_item')
    value = models.FloatField()
