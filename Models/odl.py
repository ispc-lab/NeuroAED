from __future__ import print_function
import update_x_D
import numpy as np
from numpy import linalg as LA
from utils import getCovarianceandMean
from utils import GetThreshold
from utils import normc
import time





class OnlineDictionaryLearning(object):
    """
    Solving the optimization problem:
        (D, X) = arg min_{D, X} 0.5||Y - DX||_F^2 + lamb||X||_1
    """
    def __init__(self,dataset, k,updateD_iters = 100, updateX_iters = 100,method='rand'):
        
        self.train_data=dataset['train_data'].T
        self.train_labels = dataset['train_labels']
        self.test_data=dataset['test_data'].T
        self.test_labels = dataset['test_labels']

       

        self.seed=1
        self.method=method

        self.lambd = 0.001
        self.k = k
        self.D = None
        self.init_dictionary()

        self.X = None
        self.updateD_iters = updateD_iters
        self.updateX_iters = updateX_iters

        self.threshold=None
        self.mean=None
        self.covMatInv=None

        self.err={}
        self.residual=None
        self.err['train']=None
        self.err['test']=None

        self.distance={}
        self.distance['test']=None
        self.distance['train']=None

    def init_dictionary(self):
    #''' Functon for dictionary initialization'''
        Y=self.train_data
        n_subj=Y.shape[1]
        np.random.seed(self.seed)
        if self.method == 'data':
            rand_vect = np.arange(n_subj)
            np.random.shuffle(rand_vect)
            k_rand = rand_vect[:self.k]
            self.D =Y[:,k_rand]

        elif self.method == 'rand':
            self.D = np.random.randn(Y.shape[0],self.k)
        else:
            raise ValueError('Unrecognized method!!')


    def train(self,verbose = False):
        """
        Y: numpy data [n_features, n_samples]
        k: interger: number of atoms in the dictionary
            if k is None, select k = round(0.2*n_samples)
        """
        Y=self.train_data
        n_subj=Y.shape[1]
        err=np.zeros((n_subj))

        D=self.D
        X = np.zeros((D.shape[1], Y.shape[1]))

        for it in range(self.updateX_iters):
            # update X
            lasso = update_x_D.Lasso(self.D, self.lambd)
            lasso.fit(Y, Xinit = X)
            X = lasso.coef_
            # update D
            F = np.dot(X, X.T)
            E = np.dot(Y, X.T)
            D = update_x_D.ODL_updateD(self.D, E, F, iterations = self.updateD_iters)

            residual = 0.5*update_x_D.normF2(self.train_data - np.dot(D, X)) + self.lambd*update_x_D.norm1(X)
        

        for p in range(n_subj):
            x = X[:,p]
            y=Y[:,p]
            err[p]=np.linalg.norm(np.dot(D,x) - y)
		



        distances, threshold, mean, covMatInv =GetThreshold(X.transpose())
        self.D=D
        self.X=X
        self.threshold=threshold
        self.mean=mean
        self.covMatInv=covMatInv
        self.err['train']=err
        self.distance['train']=distances
        self.residual=residual

    def predict(self, test_data=None,test_labels=None, verbose=True):
        
        if test_data is not None:
            self.test_data=test_data
            self.test_labels=test_labels

        Y=self.test_data
        n_subj=Y.shape[1]
        k = self.k
        D = self.D
        lambd=self.lambd
    
        mean=self.mean
        covMatInv=self.covMatInv
        threshold=self.threshold

        err=np.zeros((n_subj))
        distances=np.zeros((n_subj))
        labels=np.zeros((n_subj))

        X = np.zeros((D.shape[1], Y.shape[1]))
        lasso = update_x_D.Lasso(D, self.lambd)

        start_time = time.time()#开始时间
        lasso.fit(Y, Xinit = X)
        X = lasso.coef_

        for i in range(n_subj):
            x = X[:,i]
            y=Y[:,i]
            err[i]=np.linalg.norm(np.dot(D,x) - y)

            mahalanobisDist = np.dot(np.dot((x.transpose()- mean), covMatInv),(x.transpose() - mean).transpose())

            if mahalanobisDist > threshold:
                labels[i]=1

            distances[i]=mahalanobisDist
        end_time = time.time()
        test_time = end_time-start_time
        # self.test_frames =len(self.test_labels.reshape(-1,1))
        self.test_frames = 100
        nframes = self.test_frames
        fps =nframes/test_time
        time_data = self.proposal2json(start_time, end_time, test_time,nframes,fps)

        self.err['test']=err
        self.distance['test'] = distances

        self.pred_labels=labels
        return labels,distances,time_data





    def proposal2json(self,start_time, end_time, ntimes,nframes,fps):
        data = dict()
        data['start_time'] = start_time
        data['end_time'] = end_time
        data['nframes'] = nframes
        data['ntimes'] = ntimes
        data['fps'] = fps
        return data