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

######################################### AdaBOOST ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import base64
import io
import urllib

def adaboost_classifier(request):
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train AdaBoost Classifier
    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
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
    plt.title('AdaBoost Confusion Matrix')

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
    fig.update_layout(title='AdaBoost Feature Importance',
                      xaxis_title='Features',
                      yaxis_title='Importance')

    feature_importance_plot = fig.to_html(full_html=False)

    context = {
        'cm_image': cm_image_uri,
        'feature_importance_plot': feature_importance_plot,
        'classification_report': processed_report
    }
    return render(request, 'algorithms/adaboost_classifier.html', context)


######################################### XGBoost ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import plotly.graph_objects as go
import base64
import io
import urllib

def xgboost_classifier(request):
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Classifier
    clf = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')
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
    plt.title('XGBoost Confusion Matrix')

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
    fig.update_layout(title='XGBoost Feature Importance',
                      xaxis_title='Features',
                      yaxis_title='Importance')

    feature_importance_plot = fig.to_html(full_html=False)

    context = {
        'cm_image': cm_image_uri,
        'feature_importance_plot': feature_importance_plot,
        'classification_report': processed_report
    }
    return render(request, 'algorithms/xgboost_classifier.html', context)

######################################### Overfitting and Underfitting ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import base64
import io
import urllib

def overfitting_underfitting(request):
    # Generate synthetic data
    np.random.seed(42)
    X = np.sort(np.random.rand(100, 1) * 10, axis=0)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    # Fit different polynomial models
    models = {
        "Underfitting (Degree 1)": make_pipeline(PolynomialFeatures(1), LinearRegression()),
        "Optimal Fit (Degree 3)": make_pipeline(PolynomialFeatures(3), LinearRegression()),
        "Overfitting (Degree 10)": make_pipeline(PolynomialFeatures(10), LinearRegression())
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.squeeze(), y=y, mode='markers', name='Data Points'))

    # Train and visualize each model
    for label, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        fig.add_trace(go.Scatter(x=X.squeeze(), y=y_pred, mode='lines', name=label))

    fig.update_layout(title='Overfitting vs Underfitting',
                      xaxis_title='X',
                      yaxis_title='y')

    graph_html = fig.to_html(full_html=False)

    # Create a heatmap showing model performance (MSE)
    degrees = range(1, 11)
    train_errors = []
    test_errors = []

    # Split the data into training and test sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, label="Training Error")
    plt.plot(degrees, test_errors, label="Test Error")
    plt.xlabel("Model Complexity (Polynomial Degree)")
    plt.ylabel("Mean Squared Error")
    plt.title("Training vs Test Error (Overfitting & Underfitting)")
    plt.legend()

    # Save the error plot as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    error_plot_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    context = {
        'graph_html': graph_html,
        'error_plot_image': error_plot_image
    }
    return render(request, 'algorithms/overfitting_underfitting.html', context)

