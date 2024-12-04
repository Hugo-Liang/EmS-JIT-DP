import pickle
import math
import random
import time
import argparse

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from collections import Counter
from itertools import groupby

#import xgboost as xgb
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier


def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    #print('AUC-PR:  auc=%.3f' % ( lr_auc))
    return lr_auc


def train_and_evl(data, label, args):
    size = int(label.shape[0]*0.2)
    auc_ = []

    for i in range(5):
        idx = size * i
        X_e = data[idx:idx+size]
        y_e = label[idx:idx+size]

        X_t = np.vstack((data[:idx], data[idx+size:]))
        y_t = np.hstack((label[:idx], label[idx+size:]))


        model = LogisticRegression(max_iter=7000).fit(X_t, y_t)
        y_pred = model.predict_proba(X_e)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_e, y_score=y_pred, pos_label=1)
        auc_.append(auc(fpr, tpr))

    return np.mean(auc_)



def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    #print(df.keys())
    if args.drop:
        df = df.drop(columns=[args.drop])
    elif args.only:
        df = df[['Unnamed: 0','_id','date','bug','__'] + args.only]
        #print('new index:', df.keys())
    return df.values


def get_features(data):
    # return the features of yasu data
    return data[:, 5:]
    
    # return only LA
    #return data[:, 9]

def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()


def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data, long_commits, flag=None):

    #long_commit_ids = pickle.load(open(long_commits,'rb'))
    #print(type(long_commit_ids), len(long_commit_ids))

    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes = list()
    cnt_noexits = 0

    for i in range(0, len(ids)):
        
        ## filter long commits
        '''       
        if flag == 'test':
            if ids[i] in long_commit_ids:
                continue
        '''

        try:
            indexes.append(i)
        except FileNotFoundError:
            #print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1


    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)


def load_yasu_data(args):
    train_path_data = '../data/hand_crafted_features/{}/{}_train.csv'.format(args.project, args.data)
    test_path_data = '../data/hand_crafted_features/{}/{}_test.csv'.format(args.project, args.data)
    train, test = load_df_yasu_data(train_path_data, args.long_train_commits, 'test'), load_df_yasu_data(test_path_data,  args.long_test_commits, 'test')
    return train, test


def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    
    return acc, prc, rc, f1, auc_



def balance_pos_neg_in_training(X_train,y_train):

    #print(sorted(Counter(y_train).items()))
    
    #ros = RandomOverSampler(random_state=42)
    #sm = SMOTE(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    #print('y_train',type(y_train))
    #X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    #X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def baseline_algorithm(train, test, algorithm, only=False):
    _, y_train, X_train = train
    _, y_test, X_test = test
    
    #scaler = preprocessing.StandardScaler().fit(X_train)

    ##over/under sample
    X_train,y_train = balance_pos_neg_in_training(X_train,y_train)
    #print(X_train[0,])
    #X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)    
    #X_train1, X_test1 = scaler.transform(X_train),scaler.transform(X_test)
    #X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    #assert(X_train.all()==X_train1.all())
    #print(X_train[0,])
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    if algorithm == 'lr':
        starttime = time.time()
        #print(X_train[0])
        #print(y_train[0])

        param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
        
        #X_train = X_train.reshape(-1,1)
        #X_test = X_test.reshape(-1,1)
  
        #model = DecisionTreeClassifier().fit(X_train, y_train)
        #model=XGBClassifier()
        #model.fit(X_train, y_train)
        #model = LogisticRegression(max_iter=7000).fit(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=5).fit(X_train, y_train) 
        
        #print('X_train',np.shape(X_train))
        endtime = time.time()
        dtime = endtime - starttime
        #print("Train Time: %.8s s" % dtime)  #显示到微秒 
        
        starttime = time.time()
        
    
        y_pred = model.predict_proba(X_test)[:, 1]
        
        
        
        endtime = time.time()
        dtime = endtime - starttime
        #print("Eval Time: " , dtime)  #显示到微秒 
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        if only and not "cross" in args.data:
            #print('Here **********')
            auc_ = train_and_evl(X_train, y_train, args)
        #print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    elif algorithm =='dbn':
        y_pred = DBN_JIT(X_train, y_train, X_test, y_test)
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        acc, prc, rc, f1 = 0, 0, 0, 0
        #print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    else:
        #print('You need to give the correct algorithm name')
        return

    return y_test, y_pred 


parser = argparse.ArgumentParser()

parser.add_argument('-project', type=str,
                    default='openstack')
parser.add_argument('-data', type=str,
                    default='k')
parser.add_argument('-algorithm', type=str,
                    default='la')
parser.add_argument('-drop', type=str,
                    default='')
parser.add_argument('-only', nargs='+',
                    default=[])
parser.add_argument('-long_commits', type=str, default= 'long_commits_ids/')

parser.add_argument('-long_test_commits', type=str) 

args = parser.parse_args()
args.long_train_commits =  args.long_commits + args.project + '_train_long_commits.pkl' 
args.long_test_commits  =  args.long_commits + args.project + '_test_long_commits.pkl'



only = False
args.algorithm = 'lr'

train, test = load_yasu_data(args)
labels, predicts = baseline_algorithm(train=train, test=test, algorithm=args.algorithm, only=only)
auc_pc_score = auc_pc(labels, predicts)
auc_roc = roc_auc_score(y_true=labels,  y_score=predicts)


import pandas as pd
df = pd.DataFrame({'label': labels, 'pred': predicts})
df.to_csv('./pred_scores/test_sim_' + args.project + '.csv', index=False, sep=',')



y_true = labels
threshs = [0.5]
for t in threshs:
        real_pred = [1 if p > t else 0 for p in predicts]
        f1_ = f1_score(y_true=y_true,  y_pred=real_pred)
        print("Threshold: {}  AUC-ROC:{}  AUC-PR:{}  F1-Score:{}  ".format(t, auc_roc, auc_pc_score, f1_))













