from django.urls import path
from . import views

urlpatterns = [
    path('', views.algorithms, name='algorithms'),
    path('simple-linear-regression/', views.simple_linear_regression, name='simple_linear_regression'),
    path('multiple-linear-regression/', views.multiple_linear_regression, name='multiple_linear_regression'),
    path('k-nearest-neighbor/', views.k_nearest_neighbor, name='k_nearest_neighbor'),
    path('logistic-regression/', views.logistic_regression, name='logistic_regression'),
    path('naive-bayes/', views.naive_bayes_classifier, name='naive_bayes'),
    path('decision-tree-classifier/', views.decision_tree_classifier, name='decision_tree_classifier'),
    path('random-forest-classifier/', views.random_forest_classifier, name='random_forest_classifier'),
    path('adaboost-classifier/', views.adaboost_classifier, name='adaboost_classifier'),
    path('xgboost-classifier/', views.xgboost_classifier, name='xgboost_classifier'),
    path('overfitting-underfitting/', views.overfitting_underfitting, name='overfitting_underfitting'),
    path('cross-validation/', views.cross_validation, name='cross_validation'),
    path('neural-network/', views.neural_network, name='neural_network'),
    path('recurrent-neural-network/', views.recurrent_neural_network, name='recurrent_neural_network'),
    path('lstm/', views.lstm_neural_network, name='lstm_neural_network'),
    path('cnn/', views.cnn_neural_network, name='cnn_neural_network'),
]

