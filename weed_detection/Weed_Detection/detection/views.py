from django.shortcuts import render
from django.views import View
import tensorflow as tf

# from joblib import load
# model = load('./savedModels/')

# Create your views here.

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact_us.html')

def prediction(request):
    if request.method == 'POST':
        upload_image = request.FILES['upload_image']

      
        return render(request, 'Prediction.html', {'result' : y_pred})
    
    return render(request, 'Prediction.html')