#good data

import numpy as np
import utils, data_def
import multiprocessing as mp

pool = mp.Pool(10)

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE

#rf

results = []
matrices = data_def.matrices_random
def rf_func(i):
    regr=RandomForestRegressor(random_state=0)
    return regr.fit(matrices[i,:,1:], matrices[i,:,0]).feature_importances_
results_pool = np.array(pool.map(rf_func, range(0,matrices.shape[0])))

for i in range(0,matrices.shape[0]):
    regr = RandomForestRegressor(max_depth=None, random_state=0)
    results.append(regr.fit(matrices[i,:,1:], matrices[i,:,0]).feature_importances_) #for i in range(0,matrices.shape[0])]
results = np.array(results)
results_sorted=np.sort(results, axis=1)
top_5 = results_sorted[:,-1:-6:-1]

plt.hist(top_5[:,0])
plt.show()

#logreg
results = []
matrices=data_def.matrices_random
binary_labels=data_def.binary_labels_random
for i in range(0,matrices.shape[0]):
    regr = LogisticRegression()
    reg_f = regr.fit(matrices[i,:,1:], binary_labels[i])
    intercept = reg_f.intercept_
    coefs = reg_f.coef_[0]
    results.append(np.concatenate([intercept, coefs])) #for i in range(0,matrices.shape[0])]
results = np.array(results)

results_sorted=np.sort(results, axis=1)
top_5 = results_sorted[:,-1:-6:-1]
bottom_5 = results_sorted[:,0:5]

plt.hist(top_5[:,0])
plt.show()
plt.hist(bottom_5[:,0])
plt.show()

#linear regression
results_linear = []
matrices=data_def.matrices_random
for i in range(0,matrices.shape[0]):
    reg = LinearRegression()
    reg_f = reg.fit(matrices[i,:,1:], matrices[i,:,0])
    intercept = np.array([reg_f.intercept_])
    coefs = reg_f.coef_
    results_linear.append(np.concatenate([intercept, coefs]))

results = np.array(results_linear)

results_sorted=np.sort(results, axis=1)
top_5 = results_sorted[:,-1:-6:-1]
bottom_5 = results_sorted[:,0,0:5]

plt.hist(top_5[:,0])
plt.show()
plt.hist(bottom_5[:,0])
plt.show()
