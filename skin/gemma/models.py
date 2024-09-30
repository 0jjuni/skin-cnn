from django.db import models

class Cosmetics(models.Model):
    product_id = models.AutoField(primary_key=True)
    product_name = models.CharField(max_length=255)
    brand_name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    image_path = models.CharField(max_length=255)
    category_id = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    cosmetics_type = models.CharField(max_length=255)
    age_type = models.CharField(max_length=255)
    skin_type = models.CharField(max_length=255)
    moisture_type = models.IntegerField()
    pigmentation_type = models.IntegerField()
    pores_type = models.IntegerField()

    class Meta:
        db_table = 'cosmetics'  # 기존 테이블 이름 지정

    def __str__(self):
        return self.product_name
