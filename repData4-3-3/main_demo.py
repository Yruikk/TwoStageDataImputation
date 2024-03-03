import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scipy.io as sio

from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun

__author__ = "Łukasz Struski"


# ____________________MAIN FUNCTION_____________________#

def genRBF(Xtr, ytr, Xte, yte, m, cov, C, gamma):
    precomputed_svm = SVC(C=C, kernel='precomputed')

    index_train = np.arange(Xtr.shape[0])
    index_test = np.arange(Xte.shape[0]) + Xtr.shape[0]
    X = np.concatenate((Xtr, Xte), axis=0)
    del Xtr, Xte

    index_train = index_train.astype(np.intc)
    index_test = index_test.astype(np.intc)

    # train
    rbf_ker = rbf.RBFkernel(m, cov, X)
    S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                   rbf_ker.S)
    S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

    train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)  # 最后得到的Ktr

    train[(train > 1) | (np.isinf(train)) | (np.isnan(train))] = 1
    precomputed_svm.fit(train, ytr)

    # test
    S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
    test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                              completeDataId_test_new, completeDataId_train_new)  # Nte x Ntr
    test[(test > 1) | (np.isinf(test)) | (np.isinf(test))] = 1
    pred_tr = precomputed_svm.predict(train)  # (Ntr,)
    acc_tr = accuracy_score(ytr, pred_tr)
    pred_te = precomputed_svm.predict(test)
    acc_te = accuracy_score(yte, pred_te)

    return acc_tr, acc_te


def main(path_dir_data):
    # if len(sys.argv) < 2:
    #     raise ValueError("Assuming write paths to dir which includes needed files")
    # else:
    #     path_dir_data = sys.argv[1]

    # parameters for SVM
    C = 1
    gamma = 1.e-3

    precomputed_svm = SVC(C=C, kernel='precomputed')

    # read data
    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    X_train = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    y_train = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    X_test = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    y_test = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')

    index_train = np.arange(X_train.shape[0])
    index_test = np.arange(X_test.shape[0]) + X_train.shape[0]
    X = np.concatenate((X_train, X_test), axis=0)
    del X_train, X_test

    index_train = index_train.astype(np.intc)
    index_test = index_test.astype(np.intc)

    # train
    rbf_ker = rbf.RBFkernel(m, cov, X)
    S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                   rbf_ker.S)
    S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

    train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)  # 最后得到的Ktr
    precomputed_svm.fit(train, y_train)

    # test
    S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
    test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                              completeDataId_test_new, completeDataId_train_new)

    y_pred = precomputed_svm.predict(test)

    print("Accuracy classification score: {:.2f}".format(accuracy_score(y_test, y_pred)))


if __name__ == "__main__":
    mList = np.array([0.3])
    repTime = 5
    repList = np.arange(1, repTime + 1)
    for miss in mList:
        acc_res = np.zeros((repTime, 2))
        param_res = np.zeros((repTime, 2))
        for i in repList:
            path_dir_data = "./missRatio=" + str(miss) + "/" + str(i)

            m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
            cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

            Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
            ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
            Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
            yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
            Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
            yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

            # 生成C和g的取值范围
            C_range = 2.0 ** np.arange(-5, 6)
            g_range = 2.0 ** np.arange(-5, 6)

            bestcv = 0
            bestC = 0
            bestg = 0
            for C in C_range:
                for g in g_range:
                    acc_tr, acc_val = genRBF(Xtr, ytr, Xte, yte, m, cov, C, g)
                    if acc_val > bestcv:
                        bestcv = acc_val
                        bestC = C
                        bestg = g
            acc_tr, acc_te = genRBF(Xtr, ytr, Xte, yte, m, cov, bestC, bestg)
            acc_res[i - 1][0] = acc_tr
            acc_res[i - 1][1] = acc_te
            param_res[i - 1][0] = bestC
            param_res[i - 1][1] = bestg
        title = "res_m=" + str(miss) + ".mat"
        sio.savemat(title, {'acc': acc_res, 'param': param_res})