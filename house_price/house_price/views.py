from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login, logout
from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
import joblib
from num2words import num2words
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from math import floor
from .models import Predicted_Price_History 
from django.urls import reverse
 



# Load the trained model
model = joblib.load('D:\HousePricePrediction\project.joblib')



from django.http import JsonResponse

def registration(request):
    if request.method == 'POST':
        # Retrieve form data
        fname = request.POST.get('first_name')
        lname = request.POST.get('last_name')
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password')
        c_pass = request.POST.get('c_password')

        # Check if username already exists
        if User.objects.filter(username=uname).exists():
            return JsonResponse({'error': 'Username is already taken'})

        # Check if passwords match
        if pass1 != c_pass:
            return JsonResponse({'error': "Your password doesn't match"})

        # Create new user
        my_user = User.objects.create_user(first_name=fname, last_name=lname, username=uname, email=email, password=pass1)
        my_user.save()
        return JsonResponse({'success': 'Register successfully'})

    return render(request, 'registration.html')

from django.contrib import messages

def Login(request):
    if request.method == 'POST':
        username = request.POST.get('uname')
        pass_1 = request.POST.get('pass1')
        user = authenticate(request, username=username, password=pass_1)
      
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            error_message = 'Username or password is incorrect'
            messages.error(request, error_message)
            return redirect('login')
        
    return render(request, 'index.html')


def logout_page(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'home.html')

@login_required(login_url='login')
def about(request):
    return render (request,'about.html')

@login_required(login_url='login')
def contact(request):
    return render(request,'contact.html')



@login_required(login_url='login')
def predict(request):
    if request.method == 'POST':
        # Retrieve input data from the form
        bedrooms = request.POST.get('bedrooms')
        bathrooms = request.POST.get('bathrooms')
        floors = request.POST.get('floors')
        parking = request.POST.get('parking')
        roadsize = request.POST.get('roadsize')
        road_type = request.POST.get('road_type')
        area = request.POST.get('area')

        # Check if any of the fields are empty or None
        if any(val is None or val == '' for val in [bedrooms, bathrooms, floors, parking, roadsize, road_type, area]):
            # Handle the error
            return render(request, 'predict.html', {'error_message': 'Please fill in all fields'})

        # Convert the values to appropriate types
        bedrooms = int(bedrooms)
        bathrooms = int(bathrooms)
        floors = int(floors)
        parking = int(parking)
        roadsize = float(roadsize)
        road_type = int(road_type)
        area = float(area)
        
        input_data = np.array([[bedrooms, bathrooms, floors, parking, roadsize, road_type, area]])
        predicted_price = model.predict(input_data)  #  model.predict() method is used for predictions

        predicted_price_history = Predicted_Price_History.objects.create(
            user=request.user,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floors=floors,
            parking=parking,
            roadsize=roadsize,
            road_type=road_type,
            area=area,
            predicted_price=predicted_price[0],  
        )

        return HttpResponseRedirect('/predict/result/')  
    return render(request, 'predict.html')


@login_required(login_url='login')
def prediction_result(request):
    latest_prediction = Predicted_Price_History.objects.filter(user=request.user).order_by('-timestamp').first()

    if latest_prediction is not None:
        predicted_price = latest_prediction.predicted_price
        predicted_price_decimal = int(predicted_price)
        predicted_price_words = num2words(floor(predicted_price), lang='en_IN')
        
        return render(request, 'result.html', {
            'predicted_price': predicted_price_decimal,
            'predicted_price_words': predicted_price_words
        })
    else:
        default_predicted_price = 0
        predicted_price_words = "Zero"
        
        return render(request, 'result.html', {
            'predicted_price': default_predicted_price,
            'predicted_price_words': predicted_price_words
        })

from django.shortcuts import render, redirect
from .models import Message
from django.contrib import messages
from django.urls import reverse

@login_required(login_url='login')
def submit_message(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        user = request.user  

        message_content = request.POST.get('message')

       
        new_message = Message.objects.create(user=user, name=name, email=email, message=message_content, subject=subject)
        new_message.save()

        messages.success(request, 'Message received successfully!') 
        return redirect(f'{reverse("contact")}?success=true')  
    return render(request, 'contact.html')

@login_required(login_url='login')
def history(request):
    road_type_mapping = {
        1: 'Soil Stabilized',
        2: 'Gravelled',
        3: 'Concrete',
        4: 'Pitch'
    }
    
    predictions = Predicted_Price_History.objects.filter(user=request.user)


    for prediction in predictions:
        prediction.road_type = road_type_mapping.get(prediction.road_type, 'Unknown')

    return render(request, 'history.html', {'predictions': predictions})



@login_required(login_url='login')
def delete_prediction(request, prediction_id):
    prediction = Predicted_Price_History.objects.get(pk=prediction_id)
    if prediction.user == request.user:
        prediction.delete()
    return redirect('history')


