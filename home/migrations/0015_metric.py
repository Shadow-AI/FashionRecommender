# Generated by Django 4.0 on 2022-03-26 08:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0014_alter_useravatarsocial_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='Metric',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('article', models.CharField(max_length=100)),
                ('recall', models.FloatField()),
                ('precision', models.FloatField()),
                ('f1', models.FloatField()),
            ],
        ),
    ]
