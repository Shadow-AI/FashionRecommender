# Generated by Django 4.0 on 2022-01-25 10:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_alter_featurevector_vector'),
    ]

    operations = [
        migrations.AlterField(
            model_name='featurevector',
            name='vector',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='imagedb',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
    ]
