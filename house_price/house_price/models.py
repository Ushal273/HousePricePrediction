from django.db import models
from django.contrib.auth.models import User

class Predicted_Price_History(models.Model):
    user =models.ForeignKey(User, on_delete=models.CASCADE)
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    floors = models.IntegerField()
    parking = models.IntegerField()
    roadsize = models.FloatField()
    road_type = models.IntegerField()
    area = models.FloatField()
    predicted_price = models.DecimalField(max_digits=10,decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"

class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    subject = models.CharField(max_length=255 ,default='')
    message = models.CharField(max_length=555)
    
    def __str__(self) -> str:
        return f"{self.user.username}- {self.name}"
    