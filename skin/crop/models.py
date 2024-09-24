from django.db import models

class CroppedFace(models.Model):
    forehead = models.ImageField(upload_to='cropped_faces/')
    left_cheek = models.ImageField(upload_to='cropped_faces/')
    right_cheek = models.ImageField(upload_to='cropped_faces/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
