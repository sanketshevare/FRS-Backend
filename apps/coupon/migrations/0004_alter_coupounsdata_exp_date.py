# Generated by Django 4.2.7 on 2024-04-11 14:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coupon', '0003_alter_coupounsdata_exp_date'),
    ]

    operations = [
        migrations.AlterField(
            model_name='coupounsdata',
            name='exp_date',
            field=models.CharField(blank=True, default='2024-04-11 14:17:59.939422', max_length=255, null=True),
        ),
    ]
