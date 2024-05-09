from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def home(request):
    return render(request, 'index.html')

def algorithms(request):
    return render(request, 'pages/algoithms-grid-view.html')

def simple_linear_regression(request):
    return render(request, 'algorithms/simple-linear-regression.html')