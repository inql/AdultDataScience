import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN


def perform_k_means(data):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data.train_normalized)
    Z = kmeans.predict(data.train_normalized)
    score = metrics.accuracy_score(data.train_classes, Z)
    print('Accuracy:{0:f}'.format(score))
    return kmeans, Z


def perform_dbscan(data):
    db_scan = DBSCAN(eps=0.15, min_samples=600).fit(data.train_normalized)
    labels = np.array(db_scan.labels_)
    unique, counts = np.unique(labels, return_counts=True)
    result = dict(zip(unique, counts))
    print('DBSCAN stats(value, count): \n')
    for k, v in result.items():
        print(str(k) + ": " + str(v))
    return labels
