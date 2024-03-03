import os
import mpi4py.MPI as MPI
import numpy as np
from main_demo import genRBF


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

missRatio = 0.2
num_train_files = 8
data_name = "australian"
res = np.zeros(8)
path_dir_data = "./" + data_name + "/missRatio=" + str(missRatio) + "/" + str(rank + 1)

m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

Xtr = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
ytr = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
Xte = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
yte = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')
Xval = np.genfromtxt(os.path.join(path_dir_data, 'val_data.txt'), dtype=float, delimiter=',')
yval = np.genfromtxt(os.path.join(path_dir_data, 'val_labels.txt'), dtype=float, delimiter=',')

# 找到矩阵A中所有全为NaN的行的索引
B = np.isnan(Xtr)  # 将A中的NaN元素置为True，非NaN元素置为False
C = np.all(B, axis=1)  # 对B的每一行进行检查，得到每行是否全为NaN的布尔型数组
nan_rows = np.where(C)[0].tolist()  # 得到所有全为NaN行的索引，并转成列表

Xtr = np.delete(Xtr, nan_rows, axis=0)
ytr = np.delete(ytr, nan_rows, axis=0)

# 生成C和g的取值范围
C_range = 2.0 ** np.arange(-5, 6)
g_range = 2.0 ** np.arange(-5, 6)

bestcv = 0
bestC = 0
bestg = 0
for C in C_range:
    for g in g_range:
        acc_tr, acc_val = genRBF(Xtr, ytr, Xval, yval, m, cov, C, g)
        print(C, g)
        if acc_val > bestcv and acc_tr > acc_val:
            bestcv = acc_val
            bestC = C
            bestg = g
acc_tr, acc_te = genRBF(Xtr, ytr, Xte, yte, m, cov, bestC, bestg)
acc_tr = acc_tr * 100
acc_te = acc_te * 100
res_acc_tr = comm.gather(acc_tr, root=0)
res_acc_te = comm.gather(acc_te, root=0)
if rank == 0:
    mat_acc = np.zeros((num_train_files, 2))
    for i in range(num_train_files):
        mat_acc[i][0] = res_acc_tr[i]
        mat_acc[i][1] = res_acc_te[i]
    mean_acc = np.mean(mat_acc, axis=0)
    std_acc = np.std(mat_acc, axis=0)
    print(missRatio, "Mean = ", mean_acc)
    print(missRatio, "Std = ", std_acc)