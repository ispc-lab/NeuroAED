import numpy as np
import random
from scipy import linalg
from sparse_code import klimaps
from utils import getCovarianceandMean
from utils import GetThreshold
from utils import normc
import time

class ksvd(object):

    def __init__(self,dataset,iteration,sparsity,stopFlag='solidError',errGoal=10,preserveDC=False):
        
        # load dataset
        # self.train_data=normc(dataset['train_data'].T)
        self.train_data=dataset['train_data'].T
        self.train_labels = dataset['train_labels']
        self.test_data=dataset['test_data'].T
        self.test_labels = dataset['test_labels']

       

        self.pred_labels=None

        # initialize
        self.k=sparsity
        self.iteration=iteration
        self.stopFlag=stopFlag  # This parameter determines how the sparse coding period converge
        self.errGoal=errGoal
        self.preserveDC=preserveDC

        self.D=None
        self.Dinv = None

        self.err={}
        self.residual=None
        self.err['test']=None
        self.err['train']=None

        self.distance={}
        self.distance['test']=None
        self.distance['train']=None

        self.threshold=None
        self.mean=None
        self.covMatInv=None

        self.dictInitialization()

    def dictInitialization(self):
        Y=self.train_data
        self.D=Y[:,:self.k]

        self.Dinv = np.linalg.pinv(self.D)
        
        norm=1/(np.sqrt(np.sum(self.D*self.D,0)))
        self.D=np.dot(self.D,np.diag(norm))
        self.D=self.D*np.tile(np.sign(self.D[0,:]),[self.D.shape[0],1])

    def train(self):
        Y=self.train_data

        D=self.D
        Dinv=self.Dinv
        n_subj=Y.shape[1]
        err=np.zeros((n_subj))

        for iterNum in range(self.iteration):
            if self.stopFlag=='limitedAtoms':
                # This part remains unfinished
                return
            elif self.stopFlag=='solidError':
                # coef=self.OMP(self.D,Y,self.errGoal)
                for p in range(Y.shape[1]):
                    y = Y[:,p]		
                    x = klimaps(y, D, Dinv, self.k, 10)
                    err[p]=np.linalg.norm(np.dot(D,x) - y)					
                    if p==0:
                        coef = np.expand_dims(x, axis=1)
                    else:
                        coef = np.append(coef,np.expand_dims(x, axis=1), axis=1)
                
                residual = np.linalg.norm(np.dot(D,coef) - Y)

            else:
                print("invalid stop flag")
                return
            replacedVector=0
            sequence=list(range(D.shape[1]))
            random.shuffle(sequence)
            for j in sequence:
                updateWord,coef,addVector=self.updateDictionary(Y,D,j,coef)
                D[:,j]=updateWord
                replacedVector+=addVector
            nonZeroCoef=np.nonzero(coef)
            nonZeroRatio=len(nonZeroCoef[0])/(coef.shape[1]*coef.shape[0])
            print("iter:%3d, average coefficient ratio is%f "%(iterNum,nonZeroRatio))
            D=self.cleanDictionary(Y,D,coef)
            Dinv = np.linalg.pinv(D)

        distances, threshold, mean, covMatInv =GetThreshold(coef.transpose())
        self.D=D
        self.Dinv=Dinv
        self.coef=coef
        self.threshold=threshold
        self.mean=mean
        self.covMatInv=covMatInv
        self.err['train']=err
        self.distance['train']=distances
            # self.residual=residual/Y.shape[1]
        self.residual=residual
        return self.D

    # def OMP(self,D,Y,errorGoal,showFlag=True):
    #     """
    #     constructing the sparse representation using Orthogonal Matching Pursuit algorithm
    #     :param dictionary: Given dictionary with dimension [n,K]
    #     :param data: Given data to be represented with dimension [n,N]
    #     :param errorGoal:
    #     :return:
    #     """
    #     if len(Y.shape)==1:
    #         n=len(Y.shape)
    #         N=1
    #     else:
    #         n,N=Y.shape
        
    #     errThresh=errorGoal*errorGoal*n
    #     #maxNumCoef=n/2
    #     coef=np.zeros([self.k,N])
    #     for i in range(N):
    #         y=Y[:,i]
    #         residual=y
    #         index=[]
    #         atomUsed=0
    #         currentRes=np.sum(residual*residual)

    #         #while currentRes>errThresh and atomUsed<maxNumCoef:
    #         while currentRes>errThresh:
    #             atomUsed+=1
    #             proj=np.dot(np.transpose(D,[1,0]),residual)  # proj in shape [K,1]
    #             maxIndex=np.argmax(proj,axis=0)
    #             index.append(maxIndex)  # index in shape (atoms,)
    #             expression=np.dot((linalg.pinv(D[:,index])),y)   # atoms*n ,n*1 -->atoms*1
    #             residual=y-np.dot(D[:,index],expression)
    #             currentRes=np.sum(residual*residual)
    #         if len(index)>0:
    #             coef[index,i]=expression
    #     return coef

    def updateDictionary(self,data,dictionary,wordToUpdate,coefMatrix,):
        """
        update one atom in the dictionary
        return:
        1)the updated word for the item in dictionary,
        2)new coefMatrix. this is done since only nonZeroEntry is updated and we wrap this step
        3)addVector or not. IF the atom selected is used by non data, we need to delete this atom and add new one
        """
        nonZeroEntry=np.nonzero(coefMatrix[wordToUpdate,:])
        nonZeroEntry=nonZeroEntry[0]
        if len(nonZeroEntry)<1:  #the word to be updated isn't used any data
            addVector=1
            errorMat=data-np.dot(dictionary,coefMatrix)
            selectAtom=data[:,np.argmax(np.sum(errorMat,0))]
            # normalization
            selectAtom=selectAtom/np.sqrt(np.sum(selectAtom*selectAtom))
            selectAtom=selectAtom*(np.sign(selectAtom[0]))
            return selectAtom,coefMatrix,addVector
        
        #???
        addVector=0
        tmpCoefMatrix=coefMatrix[:,nonZeroEntry]
        tmpCoefMatrix[wordToUpdate,:]=0
        errorMat=data[:,nonZeroEntry]-np.dot(dictionary,tmpCoefMatrix)
        U,S,V=np.linalg.svd(errorMat,full_matrices=False)  # V refers to V' in svd
        updateWord=U[:,0]
        coefMatrix[wordToUpdate,nonZeroEntry]=S[0]*V[0,:]  # first row of V'
        return updateWord,coefMatrix,addVector

    def cleanDictionary(self,data,dictionary,coefMatrix):
        T1=3
        T2=0.99
        error=np.sum(np.square(data-np.dot(dictionary,coefMatrix)),0)
        G=np.dot(np.transpose(dictionary,[1,0]),dictionary)
        G=G-np.diag(np.diag(G))
        for j in range(dictionary.shape[1]):
            if np.max(G[j,:])>T2 or np.sum(np.abs(coefMatrix[j,:])>1e-7)<=T1:
                index=np.argmax(error)
                error[index]=0
                dictionary[:,j]=data[:,index]/np.sqrt(np.sum(data[:,index]*data[:,index]))
                G = np.dot(np.transpose(dictionary, [1, 0]), dictionary)
                G = G - np.diag(np.diag(G))
        return dictionary

    def predict(self, test_data=None,test_labels=None, verbose=True):
        if test_data is not None:
            self.test_data=test_data
            self.test_labels=test_labels

        Y=self.test_data
        n_subj=Y.shape[1]
        k = self.k
        D = self.D
        Dinv = self.Dinv
        mean=self.mean
        covMatInv=self.covMatInv
        threshold=self.threshold

        err=np.zeros((n_subj))
        distances=np.zeros((n_subj))
        labels=np.zeros((n_subj))

        start_time = time.time()#start time
        for i in range(n_subj):
            y = Y[:,i]
            # x=self.OMP(self.D,y,self.errGoal)
            x= klimaps(y, D, Dinv, k, 10)
            err[i]=np.linalg.norm(np.dot(D,x) - y)
            mahalanobisDist = np.dot(np.dot((x.transpose()- mean), covMatInv),(x.transpose() - mean).transpose())
            if mahalanobisDist > threshold:
                labels[i]=1

            distances[i]=mahalanobisDist
        
        end_time = time.time()
        test_time = end_time-start_time 
        self.test_frames =len(self.test_labels.reshape(-1,1))
        nframes = self.test_frames
        fps = nframes/test_time
        time_data = self.proposal2json(start_time, end_time, test_time,nframes,fps)

        self.err['test']=err
        self.distance['test'] = distances
		
        # labels = (err>self.residual) * 1#abnormal detection


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