from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def home(request):
    return render(request, 'index.html')

def algorithms(request):
    return render(request, 'pages/algoithms-grid-view.html')

###################################### SIMPLE LINEAR REGRESSION ######################################

# views.py
import plotly.express as px
import numpy as np
import plotly.io as pio
from django.shortcuts import render

def simple_linear_regression(request):
    # Create synthetic data for demonstration
    np.random.seed(42)
    x = np.random.rand(100)
    y = 2 * x + 1 + np.random.randn(100) * 0.1

    # Create the Plotly figure
    fig = px.scatter(x=x, y=y, labels={'x': 'Independent Variable', 'y': 'Dependent Variable'},
                     title='Simple Linear Regression')
    fig.add_traces(px.line(x=np.linspace(0, 1, 100), y=2 * np.linspace(0, 1, 100) + 1).data)

    # Incresing the graph size
    fig.update_layout(
        width=800, 
        height=600,
        title={'text': 'Simple Linear Regression', 'x': 0.5, 'xanchor': 'center'},
        )
    

    # Convert Plotly figure to HTML
    graph_html = pio.to_html(fig, full_html=False)

    return render(request, 'algorithms/simple-linear-regression.html', {'graph_html': graph_html})

######################################## MULTIPLE LINEAR REGRESSION ######################################



def multiple_linear_regression(request):
    return render(request, 'algorithms/multiple_linear_regression.html')


######################################## K NEAREST NEIGHBORS #############################################

def k_nearest_neighbor(request):
    return render(request, 'algorithms/k_nearest_neighbor.html')

######################################## LOGISTIC REGRESSION ###################################################

def logistic_regression(request):
    return render(request, 'algorithms/logistic_regression.html')


######################################## NAIVE BAYES CLASSIFIER ###################################################

def naive_bayes_classifier(request):
    return render(request, 'algorithms/naive_bayes_classifier.html')