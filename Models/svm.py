import time
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from utils import normc


#from kernels import degree_kernel, weighted_degree_kernel
# from config import Configuration as Cfg
# from utils.log import AD_Log
# from utils.pickle import dump_svm, load_svm

class OneClassSVM(object):

    def __init__(self, dataset, kernel, nu,**kwargs):
        # initialize
        self.svm = None
        self.cv_svm = None
        self.loss = 'OneClassSVM'
        self.K_train = None#train data size
        self.K_test = None
        self.nu = None
        self.gamma = None
        self.initialize_svm(kernel, nu)
        # Scores and AUC

        # load dataset
        self.train_data=(normc(dataset['train_data'].T)).T
        self.train_labels = dataset['train_labels']
        self.train_timestamp = dataset['train_timestamp']
        self.train_data_n = len(self.train_labels)
        self.val_data=dataset['val_data']
        self.val_labels = dataset['val_labels']
        self.val_timestamp = dataset['val_timestamp']
        self.val_data_n = None
        self.test_data=(normc(dataset['test_data'].T)).T
        self.test_labels = dataset['test_labels']
        self.test_timestamp = dataset['test_timestamp']
        self.test_data_n = None
        


        self.diag = {}
        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}

        self.diag['train']['scores'] = np.zeros((len(self.train_timestamp), 1))#1列
        # self.diag['val']['scores'] = np.zeros((len(self.val_timestamp), 1))#1列
        self.diag['test']['scores'] = np.zeros((len(self.test_timestamp), 1))

        self.diag['train']['auc'] = np.zeros(1)
        self.diag['val']['auc'] = np.zeros(1)
        self.diag['test']['auc'] = np.zeros(1)

        self.diag['train']['acc'] = np.zeros(1)
        self.diag['val']['acc'] = np.zeros(1)
        self.diag['test']['acc'] = np.zeros(1)

        self.rho = None

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_svm(self, kernel, nu):
        self.svm_nu = nu
        self.kernel = kernel
        if self.kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            kernel = self.kernel
        else:
            kernel = 'precomputed'
        self.svm = svm.OneClassSVM(kernel=kernel, nu=self.svm_nu)
        self.cv_svm = svm.OneClassSVM(kernel=kernel, nu=self.svm_nu)

    def train(self,train_data=None, train_labels=None, GridSearch=False,**kwargs):
        if train_data is not None:
            X_train = train_data
        else:
            X_train = self.train_data
        if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
            # self.get_kernel_matrix(kernel=self.kernel, which_set='train', **kwargs)
            self.svm.fit(self.K_train)
        else:

            if GridSearch and self.kernel == 'rbf' and (self.val_labels is not None or self.test_labels is not None):
                # use grid search cross-validation to select gamma
                print("Using GridSearchCV for hyperparameter selection...")
                # sample small hold-out set from test set for hyperparameter selection. Save as val set.
                 # sample small hold-out set from test set for hyperparameter selection. Save as val set.
                if self.val_labels is None:
                    self.test_data_n = self.test_labels.shape[0]
                    n_val_set = int(0.1 * self.test_data_n)
                    n_test_out = 0
                    n_test_norm = 0
                    n_val_out = 0
                    n_val_norm = 0
                    while (n_test_out == 0) | (n_test_norm == 0) | (n_val_out == 0) | (n_val_norm ==0):
                        perm = np.random.permutation(self.test_data_n)
                        self.val_data = self.test_data[perm[:n_val_set]]
                        self.val_labels = self.test_labels[perm[:n_val_set]]
                        # only accept small test set if AUC can be computed on val and test set
                        n_test_out = np.sum(self.test_labels[perm[:n_val_set]])
                        n_test_norm = np.sum(self.test_labels[perm[:n_val_set]] == 0)
                        n_val_out = np.sum(self.test_labels[perm[n_val_set:]])
                        n_val_norm = np.sum(self.test_labels[perm[n_val_set:]] == 0)

                    self.test_data = self.test_data[perm[n_val_set:]]
                    self.test_labels = self.test_labels[perm[n_val_set:]]
                    self.val_data_n = len(self.val_labels)
                    self.test_data_n= len(self.test_labels)



                cv_auc = 0.0
                cv_acc = 0
                for nu in np.arange(0.1, 1, 0.1):
                    for gamma in np.logspace(-10, 1, num=10, base=2):#起始点、结束点、个数、基
                         # train on selected gamma
                        self.cv_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                        self.cv_svm.fit(X_train)
                        # predict on small hold-out set
                        self.predict(self.val_data,self.val_labels,which_set='val')

                         # save model if AUC on hold-out set improved
                        if (self.diag['val']['auc'][0]  > cv_auc):
                            self.svm = self.cv_svm
                            self.nu = nu
                            self.gamma = gamma
                            cv_auc = self.diag['val']['auc'][0]
                            cv_acc = self.diag['val']['acc'][0]

                # save results of best cv run
                self.diag['val']['auc'][0] = cv_auc
                self.diag['val']['acc'][0] = cv_acc

            else:
                # if rbf-kernel, re-initialize svm with gamma minimizing the
                # numerical error
                if self.kernel == 'rbf':
                    gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)#余弦相似度
                    self.svm = svm.OneClassSVM(kernel='rbf', nu=self.svm_nu, gamma=gamma)
                    # self.svm = svm.OneClassSVM(kernel='rbf', nu=self.svm_nu)
                    # self.gamma = gamma
                self.svm.fit(X_train)
                self.nu = self.svm_nu
                #self.gamma = gamma

    def predict(self, test_data=None, test_labels=None, which_set='test', **kwargs):
        assert which_set in ('train', 'val', 'test')
        if which_set == 'train':
            X = self.train_data
            y = self.train_labels
        if which_set == 'val':
            X = self.val_data
            y = self.val_labels
        if which_set == 'test':
            if test_data is not None:
                self.test_data = test_data
                self.test_labels = test_labels
            X = self.test_data
            y = self.test_labels

        # if which_set == 'test':
        #     if test_labels is None:
        #         test_labels = np.array([1 if i%5==0  else 0 for i in range(test_data.shape[0])])
        #     self.test_data = test_data
        #     self.test_labels = test_labels
        #     self.diag['test']['scores'] = np.zeros((len(self.test_labels), 1))
        #     X = self.test_data
        #     y = self.test_labels

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                # self.get_kernel_matrix(kernel=self.kernel, which_set=which_set, **kwargs)
                if which_set == 'train':
                    scores = (-1.0) * self.svm.decision_function(self.K_train)
                    y_pred = (self.svm.predict(self.K_train) == -1) * 1
                if which_set == 'test':
                    scores = (-1.0) * self.svm.decision_function(self.K_test)
                    y_pred = (self.svm.predict(self.K_test) == -1) * 1
            else:
                if which_set == "val":
                    scores = (-1.0) * self.cv_svm.decision_function(X)#样本点到分割超平面的距离
                    y_pred = (self.cv_svm.predict(X) == -1) * 1#检测出来的异常
                else:
                    scores = (-1.0) * self.svm.decision_function(X)
                    y_pred = (self.svm.predict(X) == -1) * 1

            self.diag[which_set]['scores'][:, 0] = scores.flatten()#返回一维数组

            #self.map_labels(y_pred,scores.flatten(),which_set)

            # if y is not None:
            #     self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)
            #     if sum(y) > 0:
            #         if len(np.unique(y)) == 2:
            #             auc = roc_auc_score(y, scores.flatten())
            #             self.diag[which_set]['auc'][0] = auc

        return  y_pred, scores

    def map_labels(self, pred_labels,scores,which_set='test'):
        if which_set == 'val':
            y = self.val_labels
            data_timestamp=self.val_timestamp
        if which_set == 'test':
            y = self.test_labels
            data_timestamp=self.test_timestamp
      
        timestamps=np.unique(data_timestamp)
        i=0
        for timestamp in timestamps:
            idx = data_timestamp==timestamp
            if i ==0:
                pred_results=(np.sum(pred_labels[idx])>0.0)*1
                if pred_results ==1:
                    scores_idx=pred_labels[idx]==1
                    pred_scores=np.mean(scores[idx][scores_idx])
                else:
                    scores_idx=pred_labels[idx]==0
                    pred_scores=np.mean(scores[idx][scores_idx])
            else:
                pred_results=np.append(pred_results,(np.sum(pred_labels[idx])>0.0)*1)
                if pred_results[-1]==1:
                    scores_idx=pred_labels[idx]==1
                    pred_scores=np.append(pred_scores,np.mean(scores[idx][scores_idx]))
                else:
                    scores_idx=pred_labels[idx]==0
                    pred_scores=np.append(pred_scores,np.mean(scores[idx][scores_idx]))
            i+=1
        
        if y is not None:
            auc = roc_auc_score(y, pred_scores)
            self.diag[which_set]['auc'][0] = auc
           
