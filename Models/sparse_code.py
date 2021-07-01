import numpy as np

def klimaps(y, D, Dinv, k, max_iter):
	
	alpha = np.matmul(Dinv,y)
	a = np.sort(np.abs(alpha))
	#lambda_ = 1/a[-(k+1)]
	if a[0] < 1:
		if a[0]==0:
			lambda_=1.1
		else:
			lambda_ = 1/a[0]			
	else:
		lambda_ = a[-1]

	
	for i in range(max_iter):
		# apply sparsity constraction mapping: increase sparsity
		b = 1-np.exp(-lambda_*np.abs(alpha))
		beta = np.multiply(alpha,b)
		#apply the orthogonal projection
		alpha = beta-np.matmul(Dinv,(np.matmul(D,beta)-y))
		#update the lambda coefficient
		a = np.sort(np.abs(alpha))
		#lambda_ = 1/a[-(k+1)]
		if a[0] < 1:
			if a[0]==0:
				lambda_=1.1
			else:
				lambda_ = 1/a[0]			
		else:
			lambda_ = a[-1]
		# lambda_=1.1
		if i != 0:
			if np.sum(np.abs(alpha - alpha_prev)) < 1e-2:
				break
		alpha_prev = alpha

	idx = np.argsort(np.abs(alpha))
	alpha[idx[0:-k]] = 0

	#Least Square
	non_zero = np.nonzero(alpha)[0]
	D = D[:,non_zero]
	a = np.matmul(np.linalg.pinv(D),y)
	alpha[non_zero] = a

	return alpha


def klimaps_matrix(Y, D, Dinv, k, max_iter):
	'''Vectorized version of k-limaps'''

	Alpha = np.matmul(Dinv,Y)
	m = D.shape[1]
	N = Alpha.shape[1]
	P = np.eye(m) - np.matmul(Dinv, D)
	a = -np.sort(-np.abs(Alpha), axis=0)

	Lambda = np.tile(np.divide(1,a[k,:]),(m,1))

	for i in range(max_iter):
		b = np.multiply(Alpha,np.exp(np.multiply(-Lambda,abs(Alpha))))
		Alpha = Alpha - np.matmul(P,b)
		a = -np.sort(-np.abs(Alpha), axis=0)
		Lambda = np.tile(np.divide(1,a[k,:]),(m,1))

	idx = np.argsort(-np.abs(Alpha), axis=0)
	for i in range(N):
		Alpha[idx[k:m,i],i] = 0
		Alpha[idx[0:k,i],i] = np.matmul(np.linalg.pinv(D[:,idx[0:k,i]]), Y[:,i])

	return Alpha



    

    