import numpy as np
import  training_utils
N = 100
dim = 10
x1 = np.random.rand(N,dim )
x2 = np.random.rand(N,dim )
y = np.random.rand(N)
cv_num =5
for cv_index in range(cv_num):
    x_tn, y_tn, x_ts, y_ts = training_utils.split_cv([x1, x2], y, cv_num, cv_index)
    print(x_tn[0].shape, y_tn.shape, x_ts[0].shape, y_ts.shape)


