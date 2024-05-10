from django.shortcuts import render

def algorithms(request):
    return render(request, 'pages/algoithms-grid-view.html')

###################################### SIMPLE LINEAR REGRESSION ######################################

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
matplotlib.use('Agg') 
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

######################################### Cross Validation ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import plotly.graph_objects as go
import base64
import io
import urllib

def cross_validation(request):
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Use Mean Absolute Error as the scoring metric
    scoring = 'neg_mean_absolute_error'

    # Implement K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    # Implement Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    loo_scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)

    # Convert scores to positive values for better visualization
    kfold_scores = -kfold_scores
    loo_scores = -loo_scores

    # Compute the mean scores
    kfold_mean_score = np.mean(kfold_scores)
    loo_mean_score = np.mean(loo_scores)

    # Plot K-Fold Cross-Validation Scores
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1, len(kfold_scores) + 1)), y=kfold_scores, mode='lines+markers', name='K-Fold Scores'))
    fig1.update_layout(title='K-Fold Cross-Validation Scores',
                       xaxis_title='Fold Number',
                       yaxis_title='Mean Absolute Error (MAE)')
    kfold_html = fig1.to_html(full_html=False)

    # Boxplot of Cross-Validation Scores
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[kfold_scores, loo_scores], notch=True)
    plt.xticks([0, 1], ['K-Fold', 'Leave-One-Out'])
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Cross-Validation Score Distribution')

    # Save the boxplot image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    boxplot_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    # Create a comparison graph between K-Fold and LOOCV mean scores
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Bar(
        x=['K-Fold Mean Score', 'LOOCV Mean Score'],
        y=[kfold_mean_score, loo_mean_score],
        text=[f'{kfold_mean_score:.4f}', f'{loo_mean_score:.4f}'],
        textposition='auto',
        name='Cross-Validation Mean Scores'
    ))
    comparison_fig.update_layout(title='Comparison of Mean Scores between K-Fold and LOOCV',
                                 xaxis_title='Cross-Validation Technique',
                                 yaxis_title='Mean Absolute Error (MAE)')
    comparison_html = comparison_fig.to_html(full_html=False)

    context = {
        'kfold_html': kfold_html,
        'boxplot_image': boxplot_image,
        'comparison_html': comparison_html
    }
    return render(request, 'algorithms/cross_validation.html', context)

######################################### NEURAL NETWORK  ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import base64
import io
import urllib

def neural_network(request):
    # Create a synthetic classification dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple neural network model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Plot the training history
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig1.update_layout(title='Neural Network Training History',
                       xaxis_title='Epoch',
                       yaxis_title='Accuracy')
    training_html = fig1.to_html(full_html=False)

    # Confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    confusion_matrix = np.zeros((2, 2))
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[true, pred] += 1

    fig2 = go.Figure(data=[go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues'
    )])
    fig2.update_layout(title='Confusion Matrix',
                       xaxis_title='Predicted Label',
                       yaxis_title='Actual Label')
    confusion_html = fig2.to_html(full_html=False)

    # Classification Report Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="0.2f", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    confusion_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    context = {
        'training_html': training_html,
        'confusion_html': confusion_html,
        'confusion_image': confusion_image,
        'accuracy': f'{accuracy:.4f}'
    }

    return render(request, 'algorithms/neural_network.html', context)

######################################### Recurrent Neural Network (RNN) ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import base64
import io
import urllib

# Generate synthetic time series data
def generate_time_series_data(n_samples=500, n_time_steps=10):
    """
    Generates synthetic time series data for classification.
    - `n_samples`: Number of samples
    - `n_time_steps`: Number of time steps
    Returns:
    - X: Features of shape (n_samples, n_time_steps, 1)
    - y: Labels of shape (n_samples,)
    """
    X = np.zeros((n_samples, n_time_steps, 1))
    y = np.zeros((n_samples,))

    for i in range(n_samples):
        trend = np.random.uniform(-0.5, 0.5)
        noise = np.random.normal(scale=0.1, size=n_time_steps)
        series = trend * np.arange(n_time_steps) + noise
        X[i, :, 0] = series
        y[i] = 1 if trend > 0 else 0

    return X, y

def recurrent_neural_network(request):
    # Generate synthetic time series data
    X, y = generate_time_series_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build an LSTM-based Recurrent Neural Network model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Plot the training history
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig1.update_layout(title='RNN Training History',
                       xaxis_title='Epoch',
                       yaxis_title='Accuracy')
    training_html = fig1.to_html(full_html=False)

    # Confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    confusion_matrix = np.zeros((2, 2))
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[int(true), int(pred)] += 1

    fig2 = go.Figure(data=[go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues'
    )])
    fig2.update_layout(title='Confusion Matrix',
                       xaxis_title='Predicted Label',
                       yaxis_title='Actual Label')
    confusion_html = fig2.to_html(full_html=False)

    # Classification Report Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="0.2f", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    confusion_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    context = {
        'training_html': training_html,
        'confusion_html': confusion_html,
        'confusion_image': confusion_image,
        'accuracy': f'{accuracy:.4f}'
    }

    return render(request, 'algorithms/rnn.html', context)

######################################### LONG SHORT-TERM MEMORY (LSTM) ###################################################

from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import base64
import io
import urllib

# Generate synthetic time series data
def generate_time_series_data(n_samples=500, n_time_steps=10):
    """
    Generates synthetic time series data for classification.
    - `n_samples`: Number of samples
    - `n_time_steps`: Number of time steps
    Returns:
    - X: Features of shape (n_samples, n_time_steps, 1)
    - y: Labels of shape (n_samples,)
    """
    X = np.zeros((n_samples, n_time_steps, 1))
    y = np.zeros((n_samples,))

    for i in range(n_samples):
        trend = np.random.uniform(-0.5, 0.5)
        noise = np.random.normal(scale=0.1, size=n_time_steps)
        series = trend * np.arange(n_time_steps) + noise
        X[i, :, 0] = series
        y[i] = 1 if trend > 0 else 0

    return X, y

def lstm_neural_network(request):
    # Generate synthetic time series data
    X, y = generate_time_series_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build an LSTM-based Recurrent Neural Network model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Plot the training history
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig1.add_trace(go.Scatter(x=list(range(1, 31)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig1.update_layout(title='LSTM Training History',
                       xaxis_title='Epoch',
                       yaxis_title='Accuracy')
    training_html = fig1.to_html(full_html=False)

    # Confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    confusion_matrix = np.zeros((2, 2))
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[int(true), int(pred)] += 1

    fig2 = go.Figure(data=[go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues'
    )])
    fig2.update_layout(title='Confusion Matrix',
                       xaxis_title='Predicted Label',
                       yaxis_title='Actual Label')
    confusion_html = fig2.to_html(full_html=False)

    # Classification Report Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="0.2f", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    confusion_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    context = {
        'training_html': training_html,
        'confusion_html': confusion_html,
        'confusion_image': confusion_image,
        'accuracy': f'{accuracy:.4f}'
    }

    return render(request, 'algorithms/lstm.html', context)

######################################### Convolutional Neural Network (CNN) ###################################################


from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import base64
import io
import urllib

def generate_image_data(n_samples=1000, img_size=(28, 28), n_classes=10):
    """
    Generates synthetic image data for classification.
    - `n_samples`: Number of samples
    - `img_size`: Image dimensions as a tuple (width, height)
    - `n_classes`: Number of classes
    Returns:
    - X: Features of shape (n_samples, img_size[0], img_size[1], 1)
    - y: Labels of shape (n_samples,)
    """
    X = np.random.rand(n_samples, img_size[0], img_size[1], 1)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

def cnn_neural_network(request):
    # Generate synthetic image data
    X, y = generate_image_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a Convolutional Neural Network model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Plot the training history
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig1.update_layout(title='CNN Training History',
                       xaxis_title='Epoch',
                       yaxis_title='Accuracy')
    training_html = fig1.to_html(full_html=False)

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    confusion_matrix = np.zeros((10, 10))
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[int(true), int(pred)] += 1

    fig2 = go.Figure(data=[go.Heatmap(
        z=confusion_matrix,
        x=[f'Predicted {i}' for i in range(10)],
        y=[f'Actual {i}' for i in range(10)],
        colorscale='Blues'
    )])
    fig2.update_layout(title='Confusion Matrix',
                       xaxis_title='Predicted Label',
                       yaxis_title='Actual Label')
    confusion_html = fig2.to_html(full_html=False)

    # Classification Report Image
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="0.2f", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    confusion_image = 'data:image/png;base64,' + urllib.parse.quote(string)

    context = {
        'training_html': training_html,
        'confusion_html': confusion_html,
        'confusion_image': confusion_image,
        'accuracy': f'{accuracy:.4f}'
    }

    return render(request, 'algorithms/cnn.html', context)
