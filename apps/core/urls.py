from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('algorithms/', views.algorithms, name='algorithms'),
    path('simple-linear-regression/', views.simple_linear_regression, name='simple_linear_regression'),
]
