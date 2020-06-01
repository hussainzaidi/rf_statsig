#apply lukacz method to rf
#TO DO: do I need to take the intercept into account in model averaging?
#good data

import numpy as np
import utils, data_def

import multiprocessing as mp


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE

import os

folder = "./results/rfclas_model_averaging_random_data_multithreading/"
if not os.path.exists(folder):
    os.mkdir(folder)


n=15
import itertools
feature_combos = list(itertools.product([False,True],repeat = n))
feature_combos=feature_combos[1:] #disregard all False

#ws = np.array([])
#coef_values_matrixk_modelj = np.array([])
#beta_kji = np.zeros(shape=(1,len(feature_combos),20))#len matrices, len feature_combos, len total features

matrices = data_def.matrices_random_sm
binary_labels=data_def.binary_labels_random_sm
weighted_coef_i_matrix_k = np.zeros(shape=(len(matrices),n))#len matrices, len features


'''def rf_func(i):
    regr=RandomForestRegressor(random_state=0, n_estimators=10)
    data = matrices[k,:,1:][:,feature_combos[i]]
    reg_f = regr.fit(data, response)
    features = reg_f.feature_importances_
    loglik = utils.loglik_linear(reg_f,data,response)
    aic = utils.aic(loglik,data.shape[0],data.shape[1])
    locations = np.where(feature_combos[i])[0]
    model_coefs = np.array(feature_combos[i]).astype(float)
    np.put(model_coefs, locations, features)
    return [aic, model_coefs]
'''
####multiprocessing functions
def rf_func(k, combo):
    regr=RandomForestClassifier(random_state=0, n_estimators=10)##########
    data = matrices[k,:,1:][:,combo]
    response = binary_labels[k]
    reg_f = regr.fit(data, response)
    #features = reg_f.feature_importances_
    loglik = utils.loglik_logistic(reg_f,data,response)
    aic = utils.aic(loglik,data.shape[0],data.shape[1])
    locations = np.where(combo)[0]
    model_coefs = np.array(combo).astype(float)
    np.put(model_coefs, locations, reg_f.feature_importances_)
    return [aic, model_coefs]
pool = mp.Pool(10)


for k in range(1, len(matrices)):  # len(matrices)
    print(k)
    #results_pool = np.array(pool.map(rf_func, feature_combos))
    tups = [(k,combo) for combo in feature_combos]
    results_pool = np.array(pool.starmap(rf_func, tups))
    aic_list = results_pool[:,0].astype(float)
    beta_ji = np.array(results_pool[:,1].tolist()).astype(float)
    #beta_kij = np.append(beta_kij, beta_ij)
    #beta_kji[k]=beta_ji
    aic_min = np.min(aic_list)
    deltas = aic_list - aic_min
    #w_models = np.array([ utils.weight_modelj(deltas,l) for l in range(deltas.shape[0])])
    #print("calculating weights...")
    w_models = utils.weight_models(deltas)
    #print("averaging betas with model weights...")
    weighted_coefs = np.einsum("ji,j",beta_ji,w_models)
    print("saving the averaged betas...")
    weighted_coef_i_matrix_k[k] = weighted_coefs
    #ws = np.append(ws, w_matrixj)
    #np.save("./results/beta_"+str(k)+"_",np.round(beta_ji,3))
    #np.save(folder+"w_models_"+str(k)+"_",w_models)
    np.save(folder+"weighted_coefs_"+str(k)+"_", weighted_coefs)
np.save(folder+"all_weighted_coefs_"+str(k)+"_", weighted_coef_i_matrix_k)
#mean = 0.147, so slightly better than picking the top feature from a single model

#bottom_feature = weighted_coef_i_matrix_k.min(axis=1)
#plt.hist(bottom_feature)
#plt.show()

top_feature = weighted_coef_i_matrix_k.max(axis=1)
plt.hist(top_feature)
plt.show()
