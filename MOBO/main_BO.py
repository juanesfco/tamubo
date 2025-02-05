# -*- coding: utf-8 -*-
"""
Created on Aug 09 2023

@author: Danial Khatamsaaz
"""


import numpy as np
import pandas as pd
from pyDOE import *
from copy import deepcopy
from gpModel import gp_model
from acquisitionFunc import knowledge_gradient , expected_improvement
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
import timeit
from multiobjective import EHVI, Pareto_finder, HV_Calc

def model(X):
    f1 = X[:,0]**2+np.exp(-X[:,1]/X[:,2])
    f2 = X[:,0]+X[:,2]
    f3 = X[:,1]/(1+X[:,2])
    f4 = np.log(X[:,3]+1)*X[:,0]
    f6 = np.sin(X[:,2])+np.cos(X[:,3])
    f5 = X[:,1]*np.sin(X[:,3])+np.exp(X[:,0])
    Y1 = (f1*f2+f2/f3+f4*f5+f6)/10
    Y2 = f2**2*f3+f4/f1+f5*f6
    return Y1,Y2

iteration=5
N_dim=4
lb1=0.01
ub1=1
lb2=0.01
ub2=1
lb3=0.01
ub3=1
lb4=0.01
ub4=1
N_test=1000
N_test_final=1000
# N_alt=960
# N_samp=10
N_training=100
reps = 2
N_obj=2
goal = np.ones([1,N_obj]) ## maximizing all objectives
ref=np.zeros([1,N_obj]) ## reference point
# normal=[ub1-lb1,ub2-lb2,ub3-lb3,ub4-lb4,ub5-lb5]
# training=pd.DataFrame(pd.read_csv('/scratch/user/danialkh26/AnkitBO/training_filtered.csv', header=None)).to_numpy()

hv_curr=np.zeros([reps,iteration+1])

for j in range(reps):
    pd.DataFrame(np.array(j).reshape(1,1)).to_csv("current_rep.csv", header=None, index=None)

    x_init = lhs(N_dim,N_training)*0.99+0.01
    # x_init = training[0:N_training,0:5].reshape(N_training,N_dim)
    y_init1,y_init2 = model(x_init)
    
    
    sf1=19**2 ### variance of the GP where no obsevation exists
    sn1=np.ones([N_training])*0.01
    l1=np.array([0.15,0.15,0.15,0.15])
    GPR1=gp_model(x_init, y_init1.reshape(x_init.shape[0]), l1, sf1, sn1, N_dim, 'SE' , mean=0)
    
    sf2=30**2 ### variance of the GP where no obsevation exists
    sn2=np.ones([N_training])*0.01
    l2=np.array([0.15,0.15,0.15,0.15])
    GPR2=gp_model(x_init, y_init2.reshape(x_init.shape[0]), l2, sf2, sn2, N_dim, 'SE' , mean=0)
    
    
    y=np.concatenate((y_init1.reshape(-1,1),y_init2.reshape(-1,1)),axis=1)
    train_y=y
    y_pareto_curr,index=Pareto_finder(train_y,goal)
    hv_curr[j,0] = (HV_Calc(goal,ref,y_pareto_curr))[0]
    
    itr=0
    
    
    while itr<iteration:
    
        itr=itr+1
        
        x_test=lhs(N_dim,N_test)*0.99+0.01
        y1,var1=GPR1.predict_var(x_test)
        sig1=abs(var1)**0.5
        y2,var2=GPR2.predict_var(x_test)
        sig2=abs(var2)**0.5
        
        y=np.concatenate((y1.reshape(-1,1),y2.reshape(-1,1)),axis=1)
        sig=np.concatenate((sig1.reshape(-1,1),sig2.reshape(-1,1)),axis=1)
        
        n_jobs=multiprocessing.cpu_count()
        def calc(ii):
            e = EHVI(y[ii].reshape(1,-1),sig[ii].reshape(1,-1),goal,ref,y_pareto_curr)
            return e

        Ehvi=Parallel(n_jobs)(delayed(calc)(np.array([jj])) for jj in range(N_test))
        
        ### neutral acquisition function
        Ehvi=np.array(Ehvi)
        
        x_star=np.argmax(Ehvi)
        
        x_query=np.array(x_test[x_star]) ### here are the suggested parameters for this iteration
        # y_query=pd.DataFrame(pd.read_csv('/scratch/user/danialkh26/AnkitBO/error.csv', header=None)).to_numpy() # read error
        y_query1,y_query2=model(x_query.reshape(1,N_dim))
        
        GPR1.update(x_query.reshape(1,N_dim), y_query1.reshape(1), np.mean(sn1).reshape(1)) ## add the new data to the model
        GPR2.update(x_query.reshape(1,N_dim), y_query2.reshape(1), np.mean(sn2).reshape(1)) ## add the new data to the model
    
        train_y=np.concatenate((GPR1.ytrain().reshape(-1,1),GPR2.ytrain().reshape(-1,1)),axis=1)
        y_pareto_curr,index=Pareto_finder(train_y,goal)
        hv_curr[j,itr] = (HV_Calc(goal,ref,y_pareto_curr))[0]

    # y_max_total.append(y_max_found)
    # pd.DataFrame(y_max_total).to_csv("y_average_BO.csv", header=None, index=None)
    # x_max_total.append(x_max_found)
    
Y=np.array(hv_curr)
YY=np.mean(Y,axis=0)
SS=np.std(Y,axis=0)
pd.DataFrame(hv_curr).to_csv("hv_curr_rand.csv", header=None, index=None)
pd.DataFrame(YY).to_csv("hv_average_rand.csv", header=None, index=None)
pd.DataFrame(SS).to_csv("hv_stdv_rand.csv", header=None, index=None)
# plt.plot(YY)

# plt.plot(GPR1.ytrain(),GPR2.ytrain(),'o')
# plt.plot(y_pareto_curr[:,0],y_pareto_curr[:,1],'o')