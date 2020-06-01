import numpy as np
#import math
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor


def loglik_linear(model_f,x,y):
    preds = model_f.predict(x)
    sigma_sq = np.mean((y-preds)**2)
    loglikelihood = -x.shape[0]*(np.log(2*sigma_sq*np.pi)+1)/2
    return loglikelihood

def aic(loglikelihood, N, K):
    return -2*loglikelihood +2*K +2*K*(K+1)/(N-K-1)

def weight_models(deltas):
    norm = np.sum(np.exp(-deltas/2))
    #return math.exp(-deltas[j]/2)/np.sum(math.exp(deltas))
    return np.exp(-deltas/2)/norm


def loglik_logistic(model,data,response):
    preds = model.predict_proba(data)[:,0] #taking first element, since second element is just 1-first element
    return -log_loss(response, preds)
    #preds[preds==0]=0.001
    #preds[preds==1]=0.999
    #return np.sum(response*np.log(preds) +(1-response)*np.log(1-preds))
    
def loglik_logistic_nomodel(pred_for_loglik,response_for_loglik):
    preds = pred_for_loglik[:,0]
    #return -log_loss(response_for_loglik, preds)
    preds[preds==0]=0.001
    preds[preds==1]=0.999
    return np.sum(response_for_loglik*np.log(preds) +(1-response_for_loglik)*np.log(1-preds))

