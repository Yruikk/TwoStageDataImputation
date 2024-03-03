import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scipy.io as sio

from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import load_iris


# X, y = load_iris(return_X_y=True)
# clf = HistGradientBoostingClassifier().fit(X, y)
# print(clf.score(X, y))

data_name = 'australian'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    # path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))


data_name = 'german'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    # path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))

data_name = 'heart'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    # path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))


data_name = 'pima'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    # path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))


data_name = 'horse'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    # path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))


data_name = 'cylinder'
miss = 0.8
repTime = 8
repList = np.arange(1, repTime + 1)

result_tr = np.zeros((repTime, 1))
result_te = np.zeros((repTime, 1))
for i in repList:
    # path_dir_data = data_name + '/missRatio=' + str(miss) + "/" + str(i)
    path_dir_data = data_name + "/" + str(i)

    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
    Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
    yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

    clf = HistGradientBoostingClassifier().fit(Xtr, ytr)

    result_tr[i-1, 0] = clf.score(Xtr, ytr)
    result_te[i-1, 0] = clf.score(Xte, yte)

print(np.mean(result_te))
print(np.std(result_te))
