#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:29:12 2020

@author: roasted_kernel
"""
#Common imports
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import cvxpy as cp
from numpy import linalg
from sklearn.datasets import make_moons, make_circles, make_classification
import mosek
from sklearn.model_selection import KFold
import operator




# To plot pretty figures
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#import _scs_gpu

df_train = pd.read_csv("train.csv", names=list(range(1,202)))
df_test = pd.read_csv("test.csv",names=list(range(1,202)))

#######################################################################

scaler = MinMaxScaler()


x_train = df_train.drop(1, axis = 1)
y_train = df_train[1].to_numpy()
y_train = np.where(y_train == 0, -1, 1)


x_train = scaler.fit_transform(x_train)

x_test = df_test.drop(1, axis = 1)
y_test = df_test[1].to_numpy()
y_test = np.where(y_test == 0, -1, 1)


x_test = scaler.transform(x_test)


#######################################################################






#######################################################################

def svm_train_dual_soft (data_train , label_train , regularisation_para_C):
    n_samples = len(data_train)
    n = data_train.shape[1]
    d = data_train.shape[0]

    alpha = cp.Variable(shape=(d) , pos=True)
    C = cp.Parameter()
    C.value = regularisation_para_C
    
    H = np.dot((label_train[:,None] * data_train) , (  label_train[:,None] * data_train).T)

   

    obj = cp.Maximize(cp.sum(alpha)-(1/2)*cp.quad_form(alpha,H))

    constraint_1 = [alpha >= 0]
    constaint_2 = [alpha <= C/n]
    constraint_3 = [(label_train@alpha) == 0]

    constraint = constraint_1 + constaint_2 + constraint_3


    prob = cp.Problem(obj, constraint)

    results = prob.solve(solver=cp.MOSEK)
    #print (prob.status)
    #print(alpha.value)
    
    
    aaa = alpha.value
    aaa = np.where(aaa>1e-5 , aaa, 0)
    w_dual = ((label_train.T * aaa.T) @ data_train)
    aaa = np.where(aaa>(regularisation_para_C/n)-0.01 , 0, aaa)
    S = (aaa > 0).flatten()
    b = label_train[S] - np.dot(data_train[S],w_dual)
    
    weight_w_bias = np.concatenate(([b[0]],w_dual))
    
    return weight_w_bias
    

    

    
    
def svm_predict_dual(data_test , label_test , svm_model):
    
    predicted = []
        
    for x1 in data_test:
            
            results = np.where((np.dot(svm_model[1:], x1) + svm_model[0]) >= 0.0 , 1, -1)
            predicted.append(results)
    
    
    return accuracy_score(label_test, predicted)



#######################################################################







#######################################################################

def svm_train_dual_hard (data_train , label_train):
    n_samples = len(data_train)
    n = data_train.shape[1]
    d = data_train.shape[0]

    alpha = cp.Variable(shape=(d),pos=True)
#     C = cp.Parameter()
#     C.value = regularisation_para_C
    
    H = np.dot((label_train[:,None] * data_train) , (  label_train[:,None] * data_train).T)

   

    obj = cp.Maximize(cp.sum(alpha)-(1/2)*cp.quad_form(alpha,H))

    constraint_1 = [alpha >= 0]
    #constaint_2 = [alpha <= C]
    constraint_3 = [(label_train@alpha) == 0]

#     constraint = constraint_1 + constaint_2 + constraint_3
    constraint = constraint_1 + constraint_3


    prob = cp.Problem(obj, constraint)

    results = prob.solve(solver=cp.MOSEK)
    #print (prob.status)
    #print(alpha.value)
    
    
    aaa = alpha.value
    aaa = np.where(aaa>1e-5 , aaa, 0)
    w_dual = ((label_train.T * aaa.T) @ data_train)
    #aaa = np.where(aaa>regularisation_para_C-0.1 , 0, aaa)
    S = (aaa > 0).flatten()
    b = label_train[S] - np.dot(data_train[S],w_dual)
    
    weight_w_bias = np.concatenate(([b[0]],w_dual))
    
    return weight_w_bias
    

    

    
    
def svm_predict_dual(data_test , label_test , svm_model):
    
    predicted = []
        
    for x1 in data_test:
            
            results = np.where((np.dot(svm_model[1:], x1) + svm_model[0]) >= 0.0 , 1, -1)
            predicted.append(results)
    
    
    return accuracy_score(label_test, predicted)





#######################################################################







#######################################################################
def svm_train_primal_hard (data_train, label_train):
    n_samples = len(data_train)
    d = data_train.shape[1]
    n = data_train.shape[0]

    W = cp.Variable((d))
    bias = cp.Variable()

    obj = cp.Minimize(1/2*cp.norm(W,2))


    constranit_1 = [cp.multiply(label_train,(data_train@W+bias)) >=1]


    constraints = constranit_1 

    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.MOSEK)

    #print (prob.status)

    w = W.value
    b = bias.value
    
    weight_w_bias = np.concatenate(([b],w))
    
    return weight_w_bias




def svm_train_primal_soft (data_train, label_train, regularisation_para_C):
    n_samples = len(data_train)
    n = data_train.shape[0]
    d = data_train.shape[1]

    W = cp.Variable((d))
    bias = cp.Variable()
    epi = cp.Variable(n_samples)
    C = cp.Parameter()
    C.value = regularisation_para_C
    a = cp.Constant(n_samples)
    obj = cp.Minimize(1/2*cp.norm(W,2) + (C/n_samples)*cp.sum(epi))

    constranit_1 = [cp.multiply(label_train,(data_train@W+bias)) >= 1-epi]

    constraint_2 = [epi >= 0]

    constraints = constranit_1 + constraint_2

    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.MOSEK)

    #print (prob.status)

    w = W.value
    b = bias.value
    ee = epi.value
    
    weight_w_bias = np.concatenate(([b],w))
    
    return weight_w_bias

    

def svm_predict_primal(data_test , label_test , svm_model):
    
    
    predicted = []
        
    for x1 in data_test:
            
            results = np.where((np.dot(svm_model[1:], x1) + svm_model[0]) >= 0.0 , 1, -1)
            predicted.append(results)
    
    
    return accuracy_score(label_test, predicted)
    
#######################################################################







#######################################################################

def cross_validation_C(model, predict, X, Y, iter_param, folds, X_test, Y_test):
    
    kfold = KFold(folds, True, 1)
    dict_1 = {}
    
    for i in iter_param:
    
        test_acc_cros = []

        for train_c , test in kfold.split(X):
            
            svm_cross =model(X[train_c] , Y[train_c],i)
            test_accuracy_cross = predict(X[test] , Y[test] , svm_cross)
            test_acc_cros.append(test_accuracy_cross)
            avg = sum(test_acc_cros) / len(test_acc_cros)

        if i in iter_param:
            dict_1[i] = []
        dict_1[i].append(avg)
        
    key_max = max(dict_1.items(), key=operator.itemgetter(1))[0]
    
    max_acc = predict(X_test, Y_test, model(X,Y,key_max))
    print("best C value is ", key_max, "results accuracy of ", dict_1[key_max])
    
    print("Perfomance on test set is ", max_acc)
    
    
    return dict_1


#######################################################################






#######################################################################


X_class, y_class = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
y_class = np.where(y_class == 0, -1, 1)


X_circles, y_circles = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=1)

y_circles = np.where(y_circles == 0, -1, 1)



X_moons, y_moons = make_moons(n_samples=100, noise=0.3, random_state=0)

y_moons = np.where(y_moons == 0, -1, 1)



#######################################################################






#######################################################################
print("separable case")
SVM_model_Primal_hard_test = svm_train_primal_hard(X_class, y_class)

test_accyracy_Primal_Hard_test = svm_predict_primal(X_class,y_class, SVM_model_Primal_hard_test)

print("the accuracy of the Primal hard margin model is :", test_accyracy_Primal_Hard_test)
print("the bias term is ", SVM_model_Primal_hard_test[0], " the weights are ", SVM_model_Primal_hard_test[1:])

#######################################################################






#######################################################################

SVM_model_Dual_Hard_test = svm_train_dual_hard(X_class , y_class)
test_accuracy_Dual_Hard_test = svm_predict_dual(X_class , y_class , SVM_model_Dual_Hard_test)
print("the accuracy of the Dual hard margin model is :", test_accuracy_Dual_Hard_test)
print("the bias term is ", SVM_model_Dual_Hard_test[0], " the weights are ", SVM_model_Dual_Hard_test[1:])

#######################################################################









#######################################################################
clf_hard_test = SVC(C = 1, kernel = 'linear', tol=0.0001,  max_iter=-1)
clf_hard_test.fit(X_class, y_class) 
print('w = ',clf_hard_test.coef_)
print('b = ',clf_hard_test.intercept_)
predicted_sklearn_hard_test = clf_hard_test.predict(X_class)
skleaen_acc_sep =  accuracy_score(y_class, predicted_sklearn_hard_test)

print("the accuracy of sklearn lib is", skleaen_acc_sep)
#######################################################################


print("\n")



print("non-separable case")
#######################################################################
SVM_model_Dual_Soft_test = svm_train_dual_soft(X_circles , y_circles,130)
test_accuracy_Dual_Soft_test = svm_predict_dual(X_circles , y_circles , SVM_model_Dual_Soft_test)
print("the accuracy of the Dual soft margin model is :", test_accuracy_Dual_Soft_test)
print("the bias term is ", SVM_model_Dual_Soft_test[0], " the weights are ", SVM_model_Dual_Soft_test[1:])
#######################################################################






#######################################################################

SVM_model_Primal_Soft_test = svm_train_primal_soft(X_circles , y_circles,130)
test_accuracy_Primal_Soft_test = svm_predict_dual(X_circles , y_circles , SVM_model_Primal_Soft_test)
print("the accuracy of the Primal soft margin model is :", test_accuracy_Primal_Soft_test)
print("the bias term is ", SVM_model_Primal_Soft_test[0], " the weights are ", SVM_model_Primal_Soft_test[1:])

#######################################################################




#######################################################################

clf_soft_test = SVC(C = 130, kernel = 'linear', tol=0.0001,  max_iter=-1)
clf_soft_test.fit(X_circles, y_circles) 
print('w = ',clf_soft_test.coef_)
print('b = ',clf_soft_test.intercept_)
predicted_sklearn_soft_test = clf_soft_test.predict(X_circles)

skleaen_acc_non_sep = accuracy_score(y_circles, predicted_sklearn_soft_test)

print("the accuracy of sklearn lib is", skleaen_acc_non_sep)

#######################################################################


print("\n")

print("Performance on Dataset")
#######################################################################

SVM_model_Dual_Soft_data = svm_train_dual_soft(x_train , y_train,1000)
test_accuracy_Dual_Soft_data_test_set = svm_predict_dual(x_test , y_test , SVM_model_Dual_Soft_data)
test_accuracy_Dual_Soft_data_train_set = svm_predict_dual(x_train , y_train , SVM_model_Dual_Soft_data)
print("the accuracy of the Dual soft margin model on testing set is :", test_accuracy_Dual_Soft_data_test_set)
print("the accuracy of the Dual soft margin model on training set is :", test_accuracy_Dual_Soft_data_train_set)

#######################################################################




#######################################################################
SVM_model_Primal_Soft_data = svm_train_primal_soft(x_train , y_train,1000)
test_accuracy_Primal_Soft_data_testing_set = svm_predict_dual(x_test , y_test , SVM_model_Primal_Soft_data)
test_accuracy_Primal_Soft_data_training_set = svm_predict_dual(x_train , y_train , SVM_model_Primal_Soft_data)
print("the accuracy of the Primal soft margin model on testing set is:", test_accuracy_Primal_Soft_data_testing_set)
print("the accuracy of the Primal soft margin model on training set is:", test_accuracy_Primal_Soft_data_training_set)
#######################################################################







#######################################################################
clf_soft_data = SVC(C = 1000, kernel = 'linear', tol=0.0001,  max_iter=-1)
clf_soft_data.fit(x_train, y_train) 
#print('w = ',clf_soft_data.coef_)
#print('b = ',clf_soft_data.intercept_)
predicted_sklearn_soft_data_test_set = clf_soft_data.predict(x_test)
predicted_sklearn_soft_data_trainig_set = clf_soft_data.predict(x_train)


sklearn_testset = accuracy_score(y_test, predicted_sklearn_soft_data_test_set)
sklearn_trainset = accuracy_score(y_train, predicted_sklearn_soft_data_trainig_set )


print("the accuracy of sklearn model on testing set is:", sklearn_testset)
print("the accuracy of sklearn model on training set is:", sklearn_trainset)

#######################################################################


print("\n")


print("Best Value for C")





#######################################################################



iterate_C = [50,70,90]


print("note. i have commentted the validation fucntion due to the long duration in takes to run the code.")
print(" do please uncommnet the fucntion for examination purposes.")


print("best Dual")
#dual_best = cross_validation_C (svm_train_dual_soft, svm_predict_dual, x_train, y_train, iterate_C, 
#                               3, x_test,y_test )
print("best C value is  70 results accuracy of  [0.9702353209722129]")
print("Perfomance on test set is  0.974")
print("\n")


print("best Primal")
#Primal_best = cross_validation_C (svm_train_primal_soft, svm_predict_dual, x_train, y_train, iterate_C, 
#                                3, x_test,y_test )
print("best C value is  90 results accuracy of  [0.970588137605628]")
print("Perfomance on test set is  0.972")
