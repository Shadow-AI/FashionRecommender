from django.db import models

# Create your models here.

CLOTH_TYPE = (
    ('Dress', 'Dress'),
    ('Shirt', 'Shirt'),
    ('Pants', 'Pants'),
    ('Shorts', 'Shorts'),
    ('Shoes', 'Shoes'),
)

class ImageDB(models.Model):
    name = models.CharField(max_length=200, null=True, blank=True)
    colour = models.CharField(max_length=50)
    type = models.CharField(max_length=50, choices=CLOTH_TYPE)
    image = models.ImageField(upload_to='media')

