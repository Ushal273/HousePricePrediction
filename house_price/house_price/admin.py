# In your app's admin.py file
from django.contrib import admin
from .models import Predicted_Price_History
from .models import Message

admin.site.register(Predicted_Price_History)
admin.site.register(Message)
