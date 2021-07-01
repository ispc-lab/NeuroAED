import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os
import pickle

dictFeatureName={'0+KSVD':200,'50+KSVD':205, '100+KSVD':210,\
		'150+KSVD':215,'200+KSVD':220,\
		'250+KSVD':225,'300+KSVD':230,\
		'350+KSVD':235,'400+KSVD':240,\
		'EMST+KSVD':299,\


		'0+ODL':300,'50+ODL':305, '100+ODL':310,\
		'150+ODL':315,'200+ODL':320,\
		'250+ODL':325,'300+ODL':330,\
		'350+ODL':335,'400+ODL':340,\
			'EMST+ODL':399,\
		}

colors=['darkorange','red','blue','green','orange','peru','darkorchid','pink','purple','brown','olive','lightsalmon','cyan','indianred','firebrick','gold','yellow','deeppink',\
				'indianred','firebrick','gold','yellow','deeppink']
linestyles=[]
markers=[]


def normc(A):
	"""
	normalize each column of A to have norm2=1
	"""
	return A/np.tile(np.sqrt(np.sum(A*A, axis=0)),(A.shape[0],1))


def dump_model(model, filename=None):
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved in %s" % filename)


def load_model(filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        return pickle.load(f)


def getCovarianceandMean(samples):
    
	height, width = samples.shape#width note the pixels of each pacth, height note the number of the patch
	#Calculate the mean of each dimension
	i = 0
	j = 0
	means = np.zeros(width, float)
	sums = np.zeros(width, float)

	while(i < width):
		j = 0
		while(j < height):
			sums[i] += samples[j][i]#column
			j += 1	
		means[i] = sums[i]/height
		i += 1

	#Calculate the variance matrix
	variance = np.zeros(samples.shape, float)
	i = 0
	j = 0
	while(i < width):
		j = 0
		while(j < height):
			variance[j][i] = samples[j][i] - means[i]
			j += 1
		i += 1

	#calculate the deviation score
	varianceT = variance.transpose()
	deviation = np.zeros((width, width), float)
	deviation = np.dot(varianceT, variance)
	i = 0
	j = 0
	while(i < width):
		j = 0
		while(j < width):
			deviation[j][i] = deviation[j][i]/height
			j += 1
		i += 1

	return deviation, means.transpose()#mean for pixel at the same position of the patch, deviation between pixels in different position also call covariance

def GetThreshold(X):
    	
	covMat, mean = getCovarianceandMean(X)
	covMatInv = np.linalg.inv(covMat)

	maxThreshold = 0
	distances=[]
	for x in X:
		threshold = np.dot(np.dot((x - mean), covMatInv),(x - mean).transpose())
		distances.append(threshold)	
		if threshold > maxThreshold:
			maxThreshold = threshold

	return distances, maxThreshold, mean, covMatInv


def result_plot(resultsdir):
	dicResults={}
	for fname in os.listdir(resultsdir):
		if os.path.splitext(fname)[1]=='.txt':
			fpath = os.path.join(resultsdir, fname)

			data=np.loadtxt(fpath,delimiter=',',dtype=np.float32)
			name=list(dictFeatureName.keys())[list(dictFeatureName.values()).index(data[0,0])]

			auc_value = auc(data[:,2],data[:,1])
			# acc = 100.0 * sum(data[:,1] == data[:,3]) / len(data[:,1])
			dicResults[name]={}
			dicResults[name]['auc_value']=auc_value
			
			dicResults[name]['tpr']=data[:,1]
			dicResults[name]['fpr']=data[:,2]
			dicResults[name]['thresholds']=data[:,3]

	
		roc_plot(dicResults,resultsdir)

    
def roc_plot(dicResults,resultsdir):

    plt.figure(figsize=(10,10))
    i=0
    for name in dicResults.keys():
        plt.plot(dicResults[name]['fpr'],dicResults[name]['tpr'],color=colors[i],lw=2,label=name+' ROC curve (area = %0.3f)'  % dicResults[name]['auc_value'])
        i=i+1
 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
        
    plt.savefig(os.path.join(resultsdir,'roc_curve' + '.png'))





if __name__ == '__main__':
    result_plot('/media/autodrive/Elements SE/AED/motion/data/stairs/leijia/results/pixel_tpr_fpr/OneClassSVM')