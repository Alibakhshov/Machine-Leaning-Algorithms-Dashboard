
# Machine Learning Dashboard Website

## Introduction
Welcome to the Machine Learning Dashboard website! This platform is a comprehensive resource for understanding and visualizing machine learning algorithms. Each algorithm is presented with a detailed explanation, interactive visualizations, and practical examples.

![ML Dashboard Logo](static/images/logo.png)

## Features
- **Algorithms Covered**: A wide range of machine learning algorithms like KNN, Logistic Regression, Decision Trees, Random Forest, Neural Networks, and more.
- **Interactive Visualizations**: Utilize Plotly and other visualization libraries for interactive graphs.
- **Practical Examples**: Detailed practical code examples for each algorithm.
- **Advantages & Disadvantages**: Learn the pros and cons of each algorithm.
- **Use Cases**: Discover real-world applications.
- **Implementation Guides**: Step-by-step guides to implement each algorithm.
- **Performance Metrics**: Evaluate each algorithm using various metrics.

## Project Structure
```bash
.
├── algorithms
│   ├── templates
│   │   └── algorithms
│   │       ├── cnn.html
│   │       ├── decision-tree.html
│   │       ├── knn.html
│   │       ├── logistic-regression.html
│   │       ├── lstm.html
│   │       ├── naive-bayes.html
│   │       ├── neural-network.html
│   │       ├── random-forest.html
│   │       ├── recurrent-neural-network.html
│   │       ├── simple-linear-regression.html
│   │       └── xgboost.html
│   └── views.py
├── static
│   ├── css
│   ├── images
│   └── js
├── templates
│   ├── partials
│   └── base.html
├── urls.py
└── README.md
```

## Algorithms Covered
### 1. Simple Linear Regression

**Description:**
Simple Linear Regression is a statistical method used to understand the relationship between one independent variable and one dependent variable.

**Key Characteristics:**
- **Equation:** `y = mx + c`
- **Applications:** Predictive analysis, trend estimation, stock market analysis, sales forecasting.

### 2. K-Nearest Neighbors (KNN)

**Description:**
K-Nearest Neighbors is a simple, non-parametric algorithm used for classification and regression.

**Key Characteristics:**
- **Parameter:** `k` (number of neighbors)
- **Applications:** Image classification, fraud detection, recommender systems.

### 3. Decision Tree Classifier

**Description:**
A decision tree classifier is a predictive model that maps features to target labels using a tree-like structure.

**Key Characteristics:**
- **Parameter:** `max_depth`, `criterion`
- **Applications:** Credit scoring, disease diagnosis, customer segmentation.

### 4. Convolutional Neural Network (CNN)

**Description:**
CNNs are a type of deep learning model specifically designed for image processing and classification tasks.

**Key Characteristics:**
- **Layers:** Convolutional, pooling, fully connected.
- **Applications:** Image classification, object detection, face recognition.

### 5. Long Short-Term Memory (LSTM)

**Description:**
LSTM networks are a type of recurrent neural network capable of learning long-term dependencies in sequential data.

**Key Characteristics:**
- **Layers:** LSTM, dropout, fully connected.
- **Applications:** Stock price prediction, text generation, speech recognition.

### 6. XGBoost

**Description:**
XGBoost is an optimized gradient boosting library designed to be efficient, flexible, and portable.

**Key Characteristics:**
- **Parameters:** `n_estimators`, `max_depth`, `learning_rate`
- **Applications:** Kaggle competitions, anomaly detection, predictive maintenance.

## Getting Started
### Prerequisites
- Python 3.x
- Django
- TensorFlow
- Plotly

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Alibakhshov/ML-Algorithm-Dashboard.git
   cd MLdashboard
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server:**
   ```bash
   python manage.py runserver
   ```

### Usage
1. Open your browser and navigate to `http://127.0.0.1:8000/`
2. Explore the algorithms, interactive visualizations, and practical guides.


### Screenshots