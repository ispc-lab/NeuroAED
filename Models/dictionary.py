import numpy as np
from sparse_code import klimaps
import random
from sparse_code import klimaps_matrix
import matplotlib.pyplot as plt
from utils import getCovarianceandMean
from utils import GetThreshold
from utils import normc




class Dictionary(object):
	"""Class to handle dictionaries"""
	def __init__(self,dataset,max_iter,sparsity,method='rand'):
		self.info = 'Class for dictionary storing'
		
		# load dataset
		self.train_data=normc(dataset['train_data'].T)
		self.train_labels = dataset['train_labels']
		self.test_data=normc(dataset['test_data'].T)
		self.test_labels = dataset['test_labels']
		self.pred_labels=None


		# initialize
		self.k=sparsity
		self.method=method
		self.seed=1
		self.D=None
		self.X=None
		self.residual=None

		
		self.err={}
		self.residual=None
		self.err['train']=None
		self.err['test']=None

		self.distance={}
		self.distance['test']=None
		self.distance['train']=None

		self.D=None
		self.Dinv = None
		self.init_dictionary()
		
		self.max_iter=max_iter

		self.threshold=None
		self.mean=None
		self.covMatInv=None

	def init_dictionary(self):
    #''' Functon for dictionary initialization'''
		Y=self.train_data
		n_subj=Y.shape[1]
		np.random.seed(self.seed)
		if self.method == 'data':
			print('\nInitializing Dictionary with data...')
			rand_vect = np.arange(n_subj)
			np.random.shuffle(rand_vect)
			k_rand = rand_vect[:self.k]
			self.D =Y[:,k_rand]

		elif self.method == 'rand':
			print('\nInitializing Dictionary with random numbers...')
			self.D = np.random.randn(Y.shape[0],self.k)
		else:
			raise ValueError('Unrecognized method!!')

		
	def normalize_dict(self,D):
    	# '''Normalize dict columns to unit norm'''
		n = np.linalg.norm(D, axis=0)
		D = D / (n + np.finfo(float).eps)
		return D


	def train(self):
		'''Learn dictionaries

		Y: gallery images (features pjected in LDA space)
		N: number of subjects
		n_patch: number of patches
		max_iter: maximum number of iterations for dictionary learning
		sparsity: sparsity level (if a list of values is pecified, one dict for each sparsity level will be learned)
		seed: for random number generation
		sparse_coder: "klimaps"
		dict_learner: "LS" (Least Squares)
		n_dicts: number of dictionaries
		init_method: initialization for the dictionaries; "data" or 'rand' 
		'''
		self.D=self.normalize_dict(self.D)
		self.Dinv = np.linalg.pinv(self.D)
		D=self.D
		Dinv=self.Dinv

		vectorized = False
		dict_learner = 'LS'


		Y=self.train_data
		n_subj=Y.shape[1]
		# residual = np.zeros([self.max_iter,1])
		err=np.zeros((n_subj))

		residual=0
		Oldresidual=0

		for i in range(self.max_iter):

			print('\nIteration: ' + str(i+1) + ' ------------------------------------------------')
			print('\nComputing Sparse Codes with ' + 'klimaps' + '...')

			if vectorized: 				
				X = klimaps_matrix(Y, D, Dinv, self.k, 1)

			else:			
				for p in range(Y.shape[1]):
					y = Y[:,p]		
					x = klimaps(y, D, Dinv, self.k, 10)
					err[p]=np.linalg.norm(np.dot(D,x) - y)					
					if p==0:
						X = np.expand_dims(x, axis=1)
					else:
						X = np.append(X,np.expand_dims(x, axis=1), axis=1)
				
				if dict_learner == 'LS':
					D = np.matmul(Y,np.linalg.pinv(X))
				
			D=self.normalize_dict(D)
			Dinv = np.linalg.pinv(D)
			Oldresidual=residual
			residual = np.linalg.norm(np.dot(D,X) - Y)
			if ((Oldresidual - residual)*(Oldresidual - residual) < 1e-5):
				break	
	

		distances, threshold, mean, covMatInv =GetThreshold(X.transpose())
		self.D=D
		self.Dinv=Dinv
		self.X=X
		self.threshold=threshold
		self.mean=mean
		self.covMatInv=covMatInv
		self.err['train']=err
		self.distance['train']=distances
		self.residual=residual

	
	
	def predict(self, test_data=None,test_labels=None, verbose=True):
    
    # ''' Klimaps classification for the learned dictionary
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
	
		for i in range(n_subj):
			y = Y[:,i]
			x= klimaps(y, D, Dinv, k, 10)
			err[i]=np.linalg.norm(np.dot(D,x) - y)
			mahalanobisDist = np.dot(np.dot((x.transpose()- mean), covMatInv),(x.transpose() - mean).transpose())

			if mahalanobisDist > threshold:
				labels[i]=1

			distances[i]=mahalanobisDist
			
		self.err['test']=err
		self.distance['test'] = distances
			
		
		# labels = (err>np.max(np.unique(self.err['train']))) * 1#检测出来的异常

		self.pred_labels=labels
		return labels,distances

	def plot_dictionary_atoms(self, dictionary=None):
		"""It plots the atoms composing the dictionary.

    	Parameters
   		----------
    	dictionary: array-like, shape=(n_atoms, n_features)

   		"""
		if dictionary is None:
			dictionary=self.D
		for r in range(0, dictionary.shape[0]):
			plt.figure()
			plt.plot(dictionary[r, :])
		plt.show()

	def plot_atoms_as_histograms(self, dictionary=None):
		"""
    	It plots the atoms composing the dictionary as histograms.

    	Parameters
    	----------
    	dictionary: array_like, shape=(n_atoms, n_features)
    	"""
		if dictionary is None:
			dictionary=self.D
		for i in range(0, dictionary.shape[0]):
			fig = plt.figure()
			fig.canvas.set_window_title(str(i+1) + " atom")
			length = len(dictionary[i, :])
			x = np.asarray(range(0, length))
			w = dictionary[i, :]
			plt.hist(x, bins=length, weights=w)
			plt.xlim((0, dictionary.shape[1]))
			plt.show()	
        	

      
        
