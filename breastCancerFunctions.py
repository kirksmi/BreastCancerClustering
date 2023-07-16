# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:00:39 2021

@author: kirksmi
"""
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
import xgboost
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, LinearRegression, Lasso
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer
import scipy.stats as st
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
balanced_accuracy_score, matthews_corrcoef, precision_score, roc_auc_score,
r2_score, mean_squared_error, roc_curve, auc)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import openpyxl
from sklearn import preprocessing
import os
from sklearn.inspection import permutation_importance
import pickle
import shap
import seaborn as sns
from collections import Counter
import re
from sklearn.preprocessing import quantile_transform
from sklearn.utils import class_weight
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

# roc plot
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.ticker import FormatStrFormatter

import myFunctions



def testAlgorgithms(X, y, condition, path="./figures"):
    pltFont = {'fontname':'Arial'}

    models = []
    models.append(('LR', LogisticRegression(max_iter=5000, penalty='none')))
    # models.append(('Lasso',LogisticRegression(max_iter=250, penalty='l1', tol=1e-2, solver='saga')))  # Lasso takes very long time to train!!!
    models.append(('Ridge', LogisticRegressionCV(max_iter=5000, penalty='l2')))
    models.append(('DT', DecisionTreeClassifier(max_depth=6)))
    models.append(('AB', AdaBoostClassifier(n_estimators=150)))
    models.append(('RF', RandomForestClassifier(n_estimators=150,max_depth=6)))
    models.append(('XGB', xgboost.XGBClassifier(n_estimators=150,max_depth=6,
                                                use_label_encoder=False,
                                                eval_metric='mlogloss')))
    models.append(('BRF', BalancedRandomForestClassifier(n_estimators=100,max_depth=6)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('NC', NearestCentroid()))
    
    results = []
    names = []
    seed = 123
    scoring = 'f1_macro'
    
    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)
    
    # le = preprocessing.LabelEncoder()
    # le.fit(y)
    # y=pd.Series(le.transform(y))
    
    print("Showing average and std. dev. of F1 scores from 5-fold cross validation...")
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X, y,
                                                  cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    
    df = pd.DataFrame(results).T
    df.columns = names
    # print(df)
    loop_stats = df.describe()
    # print(loop_stats)
    
    CIs = st.t.interval(alpha=0.95, df=len(df)-1,
              loc=np.mean(df), scale=st.sem(df))
    
  
    # plot CV scores
    plt.rcParams.update(plt.rcParamsDefault) 
    # plt.rcParams['xtick.major.pad']='10'
    fig, ax = plt.subplots(figsize=(6,4))
    # ax.set_facecolor('0.9')
    bg=ax.bar(names, np.mean(df),
           yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.7)
    ax.set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    ax.set_xticklabels(names,rotation=45,**pltFont)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title('Algorithm Comparison', fontsize=20,**pltFont)
    ax.yaxis.grid(True)
    plt.ylabel("F1 Score", fontsize=18, **pltFont)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(path+'/algorithmComparison_barGraph_{}.png'.format(condition),
                bbox_inches='tight', dpi=600)
    plt.show()
    return(loop_stats)
    
def testLinearAlgorgithms(X, y, condition):
    pltFont = {'fontname':'Arial'}
    path = "./figures/"

    models = []
    models.append(('Linear', LinearRegression()))
    models.append(('Ridge', Ridge(alpha=1.0, normalize=False)))
    models.append(('Lasso', Lasso(alpha=1.0, normalize=False)))
    models.append(('XGB', xgboost.XGBRegressor(n_estimators=100,max_depth=6)))
    models.append(('RF', RandomForestRegressor(n_estimators=100,max_depth=6)))
    models.append(('AdaBoost', AdaBoostRegressor(n_estimators=100)))
    models.append(('Hist',  HistGradientBoostingRegressor()))

    results = []
    names = []
    seed = 123
    scoring = 'neg_root_mean_squared_error'
    
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
            
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X_scaled, y,
                                                  cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    
    
    df = pd.DataFrame(results).T
    df.columns = names
    print(df)
    loop_stats = df.describe()
    print(loop_stats)
    
    CIs = st.t.interval(alpha=0.95, df=len(df)-1,
              loc=np.mean(df), scale=st.sem(df))
    
    # plot CV scores
    plt.rcParams.update(plt.rcParamsDefault) 
    plt.rcParams['xtick.major.pad']='10'
    fig, ax = plt.subplots(figsize=(8,6))
    # ax.set_facecolor('0.9')
    ax.bar(names, np.mean(df),
           yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.8)
    # ax.set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    ax.set_xticklabels(names,**pltFont)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title('Algorithm Comparison', fontsize=24,**pltFont)
    ax.yaxis.grid(True)
    plt.ylabel(scoring, fontsize=20, **pltFont)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(path+'LinearComparison_barGraph_{}.png'.format(condition),
                bbox_inches='tight', dpi=600)
    plt.show()

def plot_roc(clf, X_train, y_train, X_test, y_test,
             class_names, pos_class=None,
             figsize=(8,6), show=True):
    '''
    This function is used to calculate AUC and plot the ROC curve
    for multi-class problems.

    Function Inputs:
    ----------
    1. clf:             Classifier model
    1. X_train:         Feature matrix used to train model
    2. y_train:         Target variable array used to train model
    1. X_test:          Feature matrix that model is making predictions on
    2. y_test:          Target variable array that model performance is measured against
    7. class_names:     String array of class names
    8. pos_class:       Positive class. If given, only the curve for this class will be
                        shown.
    9. fig_size:        Size of ROC curve plot

    Function Outputs: 
    ----------
    1. avg_auc:     Mean AUC calculated for all classes using the OneVsRest method.
    2. fig:         Figure object containing the ROC curve
    '''

    n_classes = len(class_names)

    if n_classes > 2:
        y_train = label_binarize(y_train, classes=[0, 1, 2])
        y_test = label_binarize(y_test, classes=[0, 1, 2])

        tpr = dict()
        classifier = OneVsRestClassifier(clf)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        # Plotting and estimation of FPR, TPR
        pltFont = {'fontname': 'Arial'}

        fpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.rcParams.update(plt.rcParamsDefault)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xticklabels(ax.get_xticks(), fontsize=14, **pltFont)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticklabels(ax.get_yticks(), fontsize=14, **pltFont)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('False Positive Rate', fontsize=16, **pltFont)
        ax.set_ylabel('True Positive Rate', fontsize=16, **pltFont)
        ax.set_title('ROC Curve', fontsize=18, **pltFont)

        if isinstance(pos_class, int):
            ax.plot(fpr[pos_class], tpr[pos_class], label='{} vs. Rest (AUC = {:.3f})'.format(
                class_names[pos_class], roc_auc[pos_class]))
        else:
            for i in range(n_classes):
                ax.plot(fpr[i], tpr[i], label='{} vs. Rest (AUC = {:.3f})'.format(
                    class_names[i], roc_auc[i]))

        ax.grid(alpha=.4)
        ax.legend(loc="lower right", fontsize=14)
        sns.despine()
        if show is True:
            plt.show()
        else:
            plt.close()

        avg_auc = np.nanmean(list(roc_auc.values()))

    else:   # binary problem
        print("ROC BINARY PROBLEM!")

    return avg_auc, fig

def tuneModels(X, y, model_name, condition, class_names, path="./tuneModels", scale="zscore", tune=True, weight=False, save=True):
    # define hyperparameters:
    AB_params = {
        'n_estimators'  : [100, 300, 300],
        'learning_rate' :  [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]}
    
    LR_params = {
        'C' : np.logspace(-3,3,10)} #np.linspace(250,750,11)} # [500, 100, 10, 1.0, 0.1, 0.01, 0.001]}
    
    RF_params = {
        "n_estimators"     : [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [3,4,6,8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}
    
    XGB_params={
        "n_estimators"     : [100, 200, 300],
        "learning_rate"    : np.linspace(0.01,0.3,5),
        "max_depth"        : [4,6,8],
        "min_child_weight" : [1, 3, 5, 7],
        "subsample"        : [0.5, 0.7, 0.9],
        "colsample_bytree" : [0.5, 0.7, 0.9],
        'gamma'            : [0, 0.01, 0.1, 0.3, 0.5]}

    SVM_params = {
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']}
    
    NC_params = {
        'shrink_threshold': np.arange(0, 1.01, 0.1)}
    
    KNN_params = {
        'n_neighbors': np.linspace(1,10,10),
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
    
    # transform y
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y=pd.Series(le.transform(y))
    
    num_class = len(y.unique())
        
    models = []
    if model_name=="AdaBoost":
        mdl = AdaBoostClassifier(random_state=123)
        param = AB_params
    if model_name=="RF":
        mdl = RandomForestClassifier(random_state=123)
        param = RF_params 
    if model_name=="BRF":
        mdl = BalancedRandomForestClassifier(random_state=123)
        param = RF_params
    if model_name=="LR":
        mdl = LogisticRegression(max_iter=5000,
                                       penalty='none', tol=1e-4)
        param = LR_params
    if model_name=="Ridge":
        mdl = LogisticRegression(max_iter=5000,
                                       penalty='l2', tol=1e-4)
        param = LR_params
    if model_name=="Lasso":
        mdl = LogisticRegression(max_iter=1000,
                                       penalty='l1', tol=1e-3, solver='saga')
        param = LR_params
    if model_name=="NC":
        mdl = NearestCentroid()
        param = NC_params
    if model_name=="XGB":
        mdl = xgboost.XGBClassifier(objective='multi:softmax',
                                               use_label_encoder=False,
                                               num_class=num_class,
                                               eval_metric='mlogloss',
                                               random_state=123)
        param = XGB_params
    if model_name=="SVM":
        mdl = SVC(random_state=123)
        param = SVM_params
    if model_name=="KNN":
        mdl = KNeighborsClassifier()
        param = KNN_params

    features = X.columns
    
    pltFont = {'fontname':'Arial'}
    # path = "./figures/tuneModels/"
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    avg = "macro"
    num_folds = 5
    
    # StratifiedKfold method
    cv = StratifiedKFold(n_splits=num_folds,
                         shuffle=True, 
                         random_state=123)
    myModels = {}
    name = model_name
    
    # for name, mdl, param in models:
    print("TRAINING {} MODEL".format(name))

    random_search = RandomizedSearchCV(mdl, param_distributions=param,
                     n_iter=5,
                     scoring='f1_macro',
                     n_jobs=-1, cv=5, verbose=3) 
    paramDict = {}

    # create empty lists to store CV scores
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    mcc_list = []
    auc_list = []
    r_list = []

    y_test = []
    y_pred = []
    cmCV = np.zeros((num_class, num_class))

    count = 0

    for train_index, test_index in cv.split(X, y):
        X_trainCV, X_testCV = X.iloc[train_index], X.iloc[test_index]
        y_trainCV, y_testCV = y.iloc[train_index], y.iloc[test_index]
        weights = class_weight.compute_sample_weight("balanced", y_trainCV)


        if scale != "none":
            if scale=="zscore":
                scaler = preprocessing.StandardScaler()
                print("PERFORMING Z-SCORE SCALING")
            elif scale=="zscore_noMean":
                scaler = preprocessing.StandardScaler(with_mean=False)
                print("PERFORMING Z-SCORE (NO MEAN) SCALING")
            elif scale=="minmax":
                scaler = preprocessing.MinMaxScaler()
                print("PERFORMING RANGE SCALING (O TO 1")
            elif scale=="quantile":
                print("PERFORMING QUANTILE SCALING")
                scaler = preprocessing.QuantileTransformer(n_quantiles=100,random_state=0)
            elif scale=="robust":
                scaler = preprocessing.RobustScaler()

            X_trainCV = pd.DataFrame(scaler.fit_transform(X_trainCV), columns=features)
            X_testCV = pd.DataFrame(scaler.transform(X_testCV), columns=features)
        else:
            print("NO FEATURE SCALING!")


        if tune==True:
            print("TUNING MODEL!")
            if weight==True:
                random_search.fit(X_trainCV, y_trainCV, sample_weight=weights)
            else:
                random_search.fit(X_trainCV, y_trainCV)
            # randomSearch_mdl = random_search.best_estimator_

            # if XGBoost, tune gamma
            # if name == "XGBoost":
            #     print("TUNING GAMMA")
            #     params_gamma = {'gamma':[0, 0.1, 0.3, 0.5]}
            #     gamma_search = GridSearchCV(estimator = randomSearch_mdl, 
            #                         param_grid = params_gamma, scoring='f1_macro',
            #                         n_jobs=-1 , cv=5)
            #     gamma_search.fit(X_trainCV, y_trainCV)
            #     best_Mdl = gamma_search.best_estimator_
            # else:
            best_Mdl = random_search.best_estimator_
        else: 
            print("NO HP TUNING!")
            if weight==True:
                best_Mdl = mdl.fit(X_trainCV, y_trainCV, sample_weight=weights)
            else:
                best_Mdl = mdl.fit(X_trainCV, y_trainCV)

        paramDict[count] = best_Mdl.get_params()
        y_predCV = best_Mdl.predict(X_testCV)

        y_test.extend(y_testCV)
        y_pred.extend(y_predCV)

        cm = confusion_matrix(y_testCV, y_predCV)
        # print("Fold {} confusion matrix: \n{} \n".format(count+1,cm))
        cmCV = cmCV+cm

        accuracy = accuracy_score(y_testCV, y_predCV)
        f1 = f1_score(y_testCV, y_predCV, average=avg)
        recall = recall_score(y_testCV, y_predCV, average=avg)
        precision = precision_score(y_testCV, y_predCV, average=avg)
        mcc = matthews_corrcoef(y_testCV, y_predCV)
        r = np.corrcoef(y_testCV, y_predCV)[0, 1]
        
        mean_auc = plot_roc(clf=best_Mdl,
                    X_train = X_trainCV, y_train = y_trainCV,
                    X_test = X_testCV, y_test = y_testCV,
                    class_names=class_names,
                    figsize=(8, 6),
                    show=False)[0]

        # assign scores to list
        acc_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        mcc_list.append(mcc)
        auc_list.append(mean_auc)
        r_list.append(r)

        count = count+1 

    # get average scores
    Accuracy = np.mean(acc_list)
    F1 = np.mean(f1_list)
    Precision = np.mean(precision_list)
    Recall = np.mean(recall_list)
    MCC = np.mean(mcc_list)
    AUC = np.mean(auc_list)
    Corr = np.mean(r_list)

    scores = [Accuracy, Recall, Precision, F1, MCC, AUC, Corr] #AUC,

    # get stats for CV scores
    loop_scores = {'Accuracy':acc_list,
                   'Recall':recall_list,
                   'Precision':precision_list,
                   'F1':f1_list,
                   'MCC':mcc_list,
                   'R':r_list,
                   'AUC':auc_list}

    df_loop_scores = pd.DataFrame(loop_scores)
    loop_stats = df_loop_scores.describe()

    CIs = st.t.interval(alpha=0.95, df=len(df_loop_scores)-1,
              loc=np.mean(df_loop_scores), scale=st.sem(df_loop_scores))
    
    
    ### plot results ###
    y_true = y_test
    cf = confusion_matrix(y_true, y_pred)
    blanks = ['' for i in range(cf.size)]
    hfont = {'fontname':'Arial'}


    group_labels = blanks
    group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in cf.flatten()/np.sum(cf)]
    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nMCC={:0.3f}\nPearson's R={:0.3f}".format(
        Accuracy,Precision,Recall,F1, MCC, Corr)
    cmap='Blues'
    
    
    # plt.figure(figsize=[50,6])
    # Plot 1: CM
    fig, axes = plt.subplots(1, 2, figsize=(10,4)) # gridspec_kw={'width_ratios': [1, 2]}
    # fig.tight_layout()
    
    # MAKE THE HEATMAP VISUALIZATION
    sns.set(font="Arial")
    sns.heatmap(cf, annot=box_labels, fmt="",
                     cmap=cmap, cbar=False,
                     ax=axes[0],
                     annot_kws={"size": 16})  #22
    axes[0].set_yticklabels(labels=class_names, rotation=90, va="center",
                       fontsize=16, **hfont)
    axes[0].set_xticklabels(labels=class_names,
                       fontsize=16, **hfont) 
    axes[0].yaxis.set_label_position("right")
    # axes[0].yaxis.set_label_coords(1.25,0.75)
    axes[0].set_ylabel(stats_text, fontsize=12, rotation=0, labelpad=50, **hfont)#labelpad=75
    
    # now plot bar graph of CV score
    axes[1].bar(df_loop_scores.columns, scores,
           yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.8)
    axes[1].set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    axes[1].set_xticks(df_loop_scores.columns)
    axes[1].set_xticklabels(df_loop_scores.columns,**pltFont,
                       rotation=45, ha="right", rotation_mode="anchor")
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('{} Classifier'.format(name), fontsize=16, **pltFont)
    axes[1].yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(path+"/{}_results.png".format(condition))
    plt.show()   

    # train final model with best params
    maxpos = mcc_list.index(max(mcc_list))
    final_params = paramDict[maxpos]
    # print("\n\nCV MCCs: {}".format(mcc_list))
    # print("\n\nBest parameters: ", final_params)
    final_Mdl = best_Mdl.set_params(**final_params)

    X_final = X.copy()
    if scale != "none":
        X_final = scaler.fit_transform(X_final)
        X_final = pd.DataFrame(X_final, columns=X.columns)

    final_Mdl.fit(X_final, y)
        
    return final_Mdl


def modelFeatureImps(models, feature_names, condition, class_names,
                     num2plot = 15, subsystems=None):

    path = "./figures/feature_importance/"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    plt.rcParams.update(plt.rcParamsDefault) 
    
    allModelImps = pd.DataFrame()
    
    for key, classifier in models.items():
        print("CURRENT MODEL: ",key)
        all_imps = None
        
        if "SVM" in key:
            continue
        
        if any(ele in key for ele in ["Regression","Lasso", "Ridge"]):
            importances = np.mean(np.absolute(classifier.coef_),axis=0)
            all_imps = classifier.coef_
        else:
            importances = classifier.feature_importances_
            
            
         # Sort feature importances in descending order
        indices = np.argsort(np.absolute(importances))[::-1]
    
         # Rearrange feature names so they match the sorted feature importances
        names = [feature_names[i] for i in indices]

        df = pd.DataFrame({'Features' : names[:num2plot],
                           'Values' : importances[indices][:num2plot]})
        
        # Create plot
        plt.figure()
        # Create plot title
        plt.title("{}: Feature Importance".format(key), fontsize=18)
        
        # Add bars
        # plt.bar(range(num2plot), importances[indices][:num2plot])  
        if subsystems is not None:
            df['Subsystem'] = [subsystems[i] for i in indices][:num2plot]
            sns.barplot(x="Values", y="Features", hue="Subsystem", data=df,
                        orient='h', dodge=False, palette="blues")
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        else:
            sns.barplot(x="Values", y="Features", data=df, orient='h',palette="Blues_r")

        # Add feature names as x-axis labels
        # plt.xticks(range(num2plot), names[:num2plot],
        #             fontsize=16, rotation=45, horizontalalignment="right")
        # plt.yticks(fontsize=16)
        plt.savefig(path+'{}_featureImportance_{}.png'.format(key, condition),
                    bbox_inches='tight', dpi=600)
        plt.show()
        
        allModelImps[key] = names[:50]
        
        # save to Excel
        feat_importance = pd.DataFrame(list(zip(feature_names, importances)),
                                              columns=['col_name','feature_importance_vals'])
        feat_importance.sort_values(by=['feature_importance_vals'],
                                        key=pd.Series.abs,
                                        ascending=False,
                                        inplace=True)
        feat_importance[0:50].to_csv('./results/{}_Top50Genes_FeatureImps_{}.csv'.format(key, condition),
              index=False) 
        
        if all_imps is not None:   # if regression model
            topGenes = pd.DataFrame()
            
            count=1
            plt.subplots(figsize=(10, 8))
            for i in classifier.classes_:
                num_class = len(classifier.classes_)
                my_imps = all_imps[i]
                current_class = class_names[i]
                
                new_ind = np.argsort(np.absolute(my_imps))[::-1]
                new_names = [feature_names[i] for i in new_ind]
                
                df_reg = pd.DataFrame({'Features' : new_names[:num2plot],
                           'Values' : (my_imps)[new_ind][:num2plot]})
        
                # get data for csv file
                topGenes[current_class] = new_names[0:50]
                topGenes[current_class+"_Values"] = my_imps[new_ind][:50]
                
                colors = []
    
                plt.subplot(num_class, 1, count)
                if subsystems is not None:
                    df_reg['Subsystem'] = [subsystems[i] for i in new_ind][:num2plot]
                    
                    sns.barplot(x="Values", y="Features", hue="Subsystem", data=df_reg,
                        orient='h', dodge=False, palette="bright")
                    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    
                else:
                    sns.barplot(x="Values", y="Features", data=df_reg, orient='h')

                # plt.yticks(np.arange(num2plot), new_names[:num2plot],
                #     fontsize=12, rotation=45, horizontalalignment="right")
                plt.xticks(fontsize=12)
                plt.xlabel("Regression Coefficient")
                # g1.set(ylabel=None)
                plt.title(current_class)
                count+=1
                
            plt.tight_layout()
            plt.savefig(path+'{}_ClassImportances_{}.png'.format(key, condition),
                    bbox_inches='tight', dpi=600)
            plt.show()
    
            topGenes.to_csv('./results/{}_FeatureImps_byClass_{}.csv'.format(key, condition),
              index=False)   
            
    return allModelImps

def quantile_normalize(df):
    """
    input: dataframe with numerical columns
    output: dataframe with quantile normalized values
    """
    df_sorted = pd.DataFrame(np.sort(df.values,
                                     axis=0), 
                             index=df.index, 
                             columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn =df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)

def modelValidation(model, X_vals, y_vals, val_IDs, condition, class_names, path="./Validation",scale="none", X_train=None, y_train=None):
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    df_ypred = []
    # if type(models) is dict 
    # for key, classifier in models.items():
    
    key = condition
    classifier = model
    print("\n {} \n".format(key))

    for X_val, y_val, dataset_ID in zip(X_vals, y_vals,
                                    val_IDs):
        # X_val = val_data.iloc[:,2:]
        features = X_val.columns
        # if key == "XGBoost":
        #     features = features.str.replace('[<,]','')
        #     classifier.get_booster().feature_names = features
        # y_val = val_data.iloc[:,1]

        le = preprocessing.LabelEncoder()
        le.fit(y_val)
        y_val=pd.Series(le.transform(y_val))

        # matchers = ['Lasso','Ridge']
        # if any(types in key for types in matchers):
        # if key != "Logistic Regression":
        # if scaleTogether==False:
        if scale=="zscore":
            scaler = preprocessing.StandardScaler()
        elif scale=="zscore_noMean":
            scaler = preprocessing.StandardScaler(with_mean=False)
        elif scale=="standardization":
            scaler = preprocessing.MinMaxScaler()
        elif scale=="quantile":
            scaler = preprocessing.QuantileTransformer(n_quantiles=500,random_state=0)
        elif scale=="robust":
            scaler = preprocessing.RobustScaler()
        else:
            print("NO SCALING!")

        if scale != "none":
            print("SCALING DATA USING {}".format(scale))
            X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=features)

        ypred = classifier.predict(X_val)
        df_ypred.append(ypred.tolist())

        if len(np.unique(y_val)) > len(np.unique(ypred)):
            unique_classes = np.unique(y_val)
            num_class = len(unique_classes)
        else:
            unique_classes = np.unique(ypred)
            num_class = len(unique_classes) 
        # print("Unique classes: ",unique_classes)

        # my_classes=[]
        # for i, c in enumerate(unique_classes):
        #     print(c)
        #     my_classes = my_classes.append(c)
        # print(my_classes)

        # print(confusion_matrix(y_val, ypred))
        plt.figure()
        make_confusion_matrix(y_val, ypred, figsize=(5,3), categories=class_names,
                                  xyplotlabels=False, cbar=False, sum_stats=True, fontsize=14)
        plt.title("{}".format(dataset_ID),fontsize=10)
        plt.savefig(path+'/Validation_ConfusionMatrix_{}_{}.png'.format(dataset_ID, condition),
              bbox_inches='tight', dpi=1200)
        plt.show()
        
        if X_train is not None and y_train is not None:
            [mean_auc, fig] = plot_roc(model,
                X_train, y_train,
                X_val, y_val,
                class_names=class_names,
                figsize=(8, 6),
                show=True)
            fig.savefig(path+"/{}_{}_ROCcurve.png".format(dataset_ID, condition),
                        bbox_inches='tight', dpi=600)
    return df_ypred
            
        
            
def LinearModelValidation(models, X_vals, y_vals, val_IDs, condition, indVar="", scale="zscore"):
    
    path = "./figures/Validation/"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
                    
        
    for key, classifier in models.items():

        print("\n {} \n".format(key))
        
        for X_val, y_val, dataset_ID in zip(X_vals, y_vals,
                                        val_IDs):
            # X_val = val_data.iloc[:,2:]
            features = X_val.columns
            # if key == "XGBoost":
            #     features = features.str.replace('[<,]','')
            #     classifier.get_booster().feature_names = features
            # y_val = val_data.iloc[:,1]
            
            # le = preprocessing.LabelEncoder()
            # le.fit(y_val)
            # y_val=pd.Series(le.transform(y_val))
            if scale=="zscore":
                scaler = preprocessing.StandardScaler().fit(X_val)
            elif scale=="robust":
                scaler = preprocessing.RobustScaler().fit(X_val)

            X_val = scaler.transform(X_val)
            X_val = pd.DataFrame(X_val,columns=features)
                
            ypred = classifier.predict(X_val)
            # print("# isNaN: ",sum(np.isnan(ypred)))
            # print("# isNaN: ",sum(np.isnan(y_val)))
            rmse = np.sqrt(metrics.mean_squared_error(y_val, ypred))
            
            hfont = {'fontname':'Arial'}


            # plt.scatter(y_val, ypred)
            f = plt.figure()
            ax = f.add_subplot(111)


            plt.text(0.9,0.9,'R2 = {:.3f}'.format(rmse), size=15, color='red',horizontalalignment='right',
                 verticalalignment='center', transform = ax.transAxes)
            sns.regplot(x=y_val, y=ypred,  line_kws={"color": "red"})
            plt.title("{}: {}".format(key, dataset_ID),fontsize=20)
            plt.xlabel("Actual {}".format(indVar),**hfont, fontsize=18)
            plt.ylabel("Predicted {}".format(indVar),**hfont, fontsize=18)
            plt.savefig(path+'{}_Validation_Scatter_{}_{}.png'.format(key,dataset_ID, condition),
                  bbox_inches='tight', dpi=600)
            plt.show()
            
def make_confusion_matrix(y_true,
                          y_pred,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(4,3),
                          cmap='Blues',
                          title=None, fontsize=28):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    cf = confusion_matrix(y_true, y_pred)
    blanks = ['' for i in range(cf.size)]
    hfont = {'fontname':'Arial'}

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.1%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        
    #if it is a binary or multi-class confusion matrix
    if len(categories)==2:
        avg = "binary"
    else:
        avg = "macro"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg)
    recall = recall_score(y_true, y_pred, average=avg)
    f1 = f1_score(y_true, y_pred, average=avg)
    mcc = matthews_corrcoef(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    
    if sum_stats:
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nMCC={:0.3f}\nPearson's R={:0.3f}".format(
            accuracy,precision,recall,f1, mcc, r)
    else:
        stats_text = ""

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False
    
    if categories == 'auto':
        categories = range(len(categories))

    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font="Arial")
    ax = sns.heatmap(cf, annot=box_labels, fmt="",
                     cmap=cmap, cbar=cbar,
                     annot_kws={"size": (fontsize)})  #22
    ax.set_yticklabels(labels=categories, rotation=90, va="center",
                       fontsize=(fontsize*0.9), **hfont)
    ax.set_xticklabels(labels=categories,
                       fontsize=(fontsize*0.9), **hfont)   # 20

    if xyplotlabels :  # show True/Predicted labels and put stats below plot
        plt.ylabel('True label', fontweight='bold', **hfont, fontsize=(fontsize*0.7))
        plt.xlabel('Predicted label' + stats_text, fontweight='bold', **hfont, fontsize=(fontsize*0.7))
    elif cbar:   # show color bar on right and stats below 
        plt.xlabel(stats_text, fontsize=(fontsize*0.5), **hfont)
    else:   # no color or True/Predicted labels, so put stats on right
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.25,0.75)
        plt.ylabel(stats_text, fontsize=(fontsize*0.7), rotation=0, **hfont, ma='left') #labelpad=75
    
    if title:
        plt.title(title, **hfont)
        
    plt.tight_layout()
    return ax


def tuneLinearModels(X, y, model_names, condition, tune=True, scale="zscore"):
    from tune_sklearn import TuneSearchCV
    # define hyperparameters:
    
    LR_params = {
        'alpha' : [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]}
    
    RF_params = {
        'max_features': ['auto', 'sqrt'],
        "n_estimators"     : [50, 100, 200, 300],
        'max_depth': [3,4,6,8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}
    
    Ada_params={
        "n_estimators"     : [50, 100, 250],
        "learning_rate"    : [0.001, 0.01, 0.1, 0.5, 1],
        "loss"             : ['linear', 'square', 'exponential']}
    
    XGB_params={
        "n_estimators"     : [50, 100, 200, 300],
        "learning_rate"    : [0.01, 0.01, 0.05, 0.1, 0.3],
        "max_depth"        : [2,4,6,8,10],
        "min_child_weight" : [3, 5, 7],
        "subsample"        : [0.7, 0.8, 0.9],
        "colsample_bytree" : np.linspace(0.3,0.8,6),
        'gamma'            : [0, 0.1, 0.3, 0.5]}
    
    hist_params={
        "max_iter"         : [100, 250, 500],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.5],
        "max_depth"        : [4,6,8,10],
        "min_samples_leaf" : [10, 15, 20],
        "l2_regularization": [0, 0.01, 0.1]}
    
    num_class = len(y.unique())

    classifier_AB = AdaBoostRegressor(random_state=123) #multi:softmax
    classifier_RF = RandomForestRegressor(n_estimators=150,random_state=123) #400
    classifier_XGB = xgboost.XGBRegressor(random_state=123)
    classifier_ridge = Ridge(normalize=False)
    classifier_hist = HistGradientBoostingRegressor()

    classifier_lasso = LogisticRegression(max_iter=1000,
                                        penalty='l1', tol=1e-4, solver='saga')
    # classifier_SVM = SVC(random_state=123)
    
    models = []
    if "AdaBoost" in model_names:
        models.append(('AdaBoost', classifier_AB, Ada_params))
    if "Random Forest" in model_names:
        models.append(('Random Forest', classifier_RF, RF_params))
    # if "Logistic Regression" in model_names:
    #     models.append(('Logistic Regression', classifier_LR, LR_params))
    if "Ridge" in model_names:
        models.append(('Ridge', classifier_ridge, LR_params))
    if "Lasso" in model_names:
        models.append(('Lasso', classifier_lasso, LR_params))
    if "XGBoost" in model_names:
        models.append(('XGBoost', classifier_XGB, XGB_params))
    if "Hist" in model_names:
        models.append(('HistGradient', classifier_hist, hist_params))
    # if "SVM" in model_names:
    #     models.append(('SVM', classifier_SVM, SVM_params))
    
    pltFont = {'fontname':'Arial'}
    path = "./figures/tuneModels/"
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    num_folds = 5
    
    # StratifiedKfold method
    cv = KFold(n_splits=num_folds,
                         shuffle=True, 
                         random_state=123)
    myModels = {}
    final_r2s = []
    final_rmses = []
    names = []
    
    for name, mdl, param in models:
        names.append(name)
        print("TRAINING {} MODEL".format(name))
        
        random_search = RandomizedSearchCV(mdl, param_distributions=param,
                          n_iter=5,
                          scoring='neg_root_mean_squared_error',
                          n_jobs=-1, cv=5, verbose=3) 

        paramDict = {}
    
        # create empty lists to store CV scores
        r2_list = []
        rmse_list = []
        
        y_test = []
        y_pred = []
        cmCV = np.zeros((num_class, num_class))
        
        count = 0
        
        for train_index, test_index in cv.split(X, y):
            X_trainCV, X_testCV = X.iloc[train_index], X.iloc[test_index]
            y_trainCV, y_testCV = y.iloc[train_index], y.iloc[test_index]
            
            # if name != "Logistic Regression":
            print("SCALING DATA!")
            if scale=="zscore":
                scaler = preprocessing.StandardScaler().fit(X_trainCV)
            elif scale=="robust":
                scaler = preprocessing.RobustScaler().fit(X_trainCV)
            
            X_trainCV = scaler.transform(X_trainCV)
            X_testCV = scaler.transform(X_testCV)
                
            if tune==True:
                print("TUNING HPs!")

                random_search.fit(X_trainCV, y_trainCV)                
                best_Mdl = random_search.best_estimator_
            else: 
                print("NO HP TUNING!")
                best_Mdl = mdl.fit(X_trainCV, y_trainCV)
            
            paramDict[count] = best_Mdl.get_params()
            y_predCV = best_Mdl.predict(X_testCV)
            
            y_test.extend(y_testCV)
            y_pred.extend(y_predCV)
            
            # cm = confusion_matrix(y_testCV, y_predCV)
            # print("Fold {} confusion matrix: \n{} \n".format(count+1,cm))
            # cmCV = cmCV+cm
            
            r2 = r2_score(y_testCV, y_predCV)
            rmse = mean_squared_error(y_testCV, y_predCV, squared=False)
                
            # assign scores to list
            r2_list.append(r2)
            rmse_list.append(rmse)
        
            count = count+1
    
      
        # get average scores
        R2 = np.mean(r2_list)
        RMSE = np.mean(rmse_list)
        final_r2s.append(R2)
        final_rmses.append(RMSE)
        
        # scores = [R2, RMSE] #AUC,
        # df_loop_scores = pd.DataFrame(loop_scores)
        # loop_stats = df_loop_scores.describe()
        # print(loop_stats)
        # CIs = st.t.interval(alpha=0.95, df=len(df_loop_scores)-1,
        #           loc=np.mean(df_loop_scores), scale=st.sem(df_loop_scores))
              
        # train final model with best params
        maxpos = r2_list.index(max(r2_list))
        final_params = paramDict[maxpos]
        # print("CV MCCs: {}".format(mcc_list))
        print("Best parameters: ", final_params)
        final_Mdl = best_Mdl.set_params(**final_params)
        
        X_final = X.copy()
        # if name != "Logistic Regression":
        X_final = scaler.fit_transform(X_final)
        X_final = pd.DataFrame(X_final,columns=X.columns)

        final_Mdl.fit(X_final, y)
        myModels[name] = final_Mdl
        
    # ax.set_facecolor('0.9')
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(names, final_rmses,
           # yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.8)
    # ax.set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    ax.set_xticklabels(names,**pltFont)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title('Algorithm Comparison', fontsize=24,**pltFont)
    ax.yaxis.grid(True)
    plt.ylabel("RMSE", fontsize=20, **pltFont)
    plt.show()
    
    plt.tight_layout()
    plt.savefig(path+'rmse_CV_barGraph_{}.png'.format(condition),
                bbox_inches='tight', dpi=600)
    plt.show()
        
    return myModels