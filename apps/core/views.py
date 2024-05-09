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

######################################## DECISION TREE ###################################################


def decision_tree_classifier(request):
    return render(request, 'algorithms/decision_tree_classifier.html')

######################################## RANDOM FOREST ###################################################


# views.py
from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import base64
import io
import urllib

def random_forest_classifier(request):
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert nested dictionary into a more readable format for Django templates
    processed_report = {
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "accuracy": report["accuracy"]
    }

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')

    # Save the confusion matrix plot as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    cm_image_uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    # Create feature importance plot using Plotly
    feature_importances = clf.feature_importances_
    features = [f'Feature {i}' for i in range(10)]
    fig = go.Figure([go.Bar(x=features, y=feature_importances)])
    fig.update_layout(title='Random Forest Feature Importance',
                      xaxis_title='Features',
                      yaxis_title='Importance')

    feature_importance_plot = fig.to_html(full_html=False)

    context = {
        'cm_image': cm_image_uri,
        'feature_importance_plot': feature_importance_plot,
        'classification_report': processed_report
    }
    return render(request, 'algorithms/random_forest_classifier.html', context)
