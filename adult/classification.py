import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

import tensorflow as tf

import numpy as np


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def decision_tree(data):
    print('\n\n -- DECISION TREE --')
    dtc = DecisionTreeClassifier()
    dtc.fit(data.train_inputs, data.train_classes)
    y_pred = dtc.predict(data.test_inputs)
    print('Accuracy: ', dtc.score(data.test_inputs, data.test_classes))
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, y_pred))
    return dtc.score(data.test_inputs, data.test_classes), confusion_matrix(
        data.test_classes, y_pred)


def naive_bayes(data):
    print('\n\n - - NAIVE BAYES - -')
    gnb = GaussianNB()
    gnb.fit(data.train_inputs, data.train_classes)
    y_pred = gnb.predict(data.test_inputs)
    print('Accuracy: ', accuracy_score(data.test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, y_pred))
    return accuracy_score(data.test_classes, y_pred), confusion_matrix(data.test_classes,
                                                                                 y_pred)


def k_nn(data, neighbors):
    print('\n\n - - kNN (', neighbors, ') - -')
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
    knn.fit(data.train_inputs, data.train_classes)
    y_pred = knn.predict(data.test_inputs)
    print('Accuracy: ', accuracy_score(data.test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, y_pred))
    return accuracy_score(data.test_classes, y_pred), confusion_matrix(data.test_classes,
                                                                                 y_pred)


def neural_network(data):
    print('\n\n - - NEURAL NETWORK - -')

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=[112]))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit((data.train_inputs),
              (data.train_classes),
              validation_data=((data.test_inputs), (data.test_classes)),
              epochs=20,
              verbose=0)
    test_loss, test_acc = model.evaluate(data.test_inputs, data.test_classes, verbose=0)
    classes_pred = model.predict_classes(data.test_inputs, verbose=0)
    print('Accuracy: ', test_acc)
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, classes_pred))
    return test_acc, confusion_matrix(data.test_classes, classes_pred)


def random_forest(data):
    print('\n\n - - RANDOM FOREST - -')
    rf = RandomForestClassifier()
    rf.fit(data.train_inputs, data.train_classes)
    y_pred = rf.predict(data.test_inputs)
    print('Accuracy: ', accuracy_score(data.test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, y_pred))
    return accuracy_score(data.test_classes, y_pred), confusion_matrix(data.test_classes,
                                                                                 y_pred)


def perform_svm(data, svm_model):
    print('\n\n - - SVM - -')
    svm = svm_model
    svm.fit(data.train_inputs, data.train_classes)
    y_pred = svm.predict(data.test_inputs)
    print('Accuracy: ', accuracy_score(data.test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(data.test_classes, y_pred))
    return accuracy_score(data.test_classes, y_pred), confusion_matrix(data.test_classes,
                                                                                 y_pred)



class ClassificationModel:
    def __init__(self):
        self.acc = dict()
        self.conf_matrix = dict()

    def perform_all(self, data):
        self.acc['decision-tree'], self.conf_matrix['decision-tree'] = decision_tree(data)
        self.acc['k-nn1'], self.conf_matrix['k-nn1'] = k_nn(data, 1)
        self.acc['k-nn3'], self.conf_matrix['k-nn3'] = k_nn(data, 3)
        self.acc['naive-bayes'], self.conf_matrix['naive-bayes'] = naive_bayes(data)
        self.acc['random-forest'], self.conf_matrix['random-forest'] = random_forest(data)
        self.acc['neural-network'], self.conf_matrix['neural-network'] = neural_network(data)
        # self.perform_all_svm(data)

    # def perform_all_svm(self, data):
        # self.acc['svc-rbf'], self.conf_matrix['svc-rbf'] = perform_svm(data, SVC(kernel='rbf', C=1, tol=1e-3))
        # self.acc['svc-poly'], self.conf_matrix['svc-poly'] = perform_svm(data, SVC(kernel='poly'))
        # self.acc['svc-sigmoid'], self.conf_matrix['svc-sigmoid'] = perform_svm(data, SVC(kernel='sigmoid'))
