# Generated by Django 4.2.7 on 2024-01-23 10:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coupon', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='coupounsdata',
            name='code',
            field=models.CharField(max_length=255, unique=True),
        ),
        migrations.AlterField(
            model_name='coupounsdata',
            name='exp_date',
            field=models.CharField(blank=True, default='2024-01-23 10:12:22.148486', max_length=255, null=True),
        ),
    ]
