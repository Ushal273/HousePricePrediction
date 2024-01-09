from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.registration, name='registration'),
    path('login/', views.Login, name='login'),  
    path('home/', views.home, name='home'),
    path('contact/',views.contact, name='contact'),
    path('submit_message/', views.submit_message, name='submit_message'),
    path('history/', views.history, name='history'),
    path('about/',views.about, name='about'),
    path('logout/', views.logout_page, name='logout'),
    path('predict/', views.predict, name='predict'),
    path('predict/result/', views.prediction_result, name='prediction_result'),
     path('delete_prediction/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
]
