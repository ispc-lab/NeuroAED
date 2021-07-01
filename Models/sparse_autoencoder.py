import numpy as np
from numpy.random import randint, rand
import scipy.io as scio 
from scipy.optimize import minimize
import time
import os, os.path

IMAGE_FILE = "IMAGES.mat"
IMAGE_MAT_TAG = 'IMAGES'
PATCH_SIZE = (14, 18)
PATCH_SIZE_1D = PATCH_SIZE[0] * PATCH_SIZE[1]
N_PATCHES_TRAIN = 10000



def sample_training_images(whitening=True, recenter_std=3.0):
    # Load the image. Here it is a 512*512*10 numpy array.
    test_dir = os.path.join(data_dir,'abnor')
    all_images = scio.loadmat(IMAGE_FILE)[IMAGE_MAT_TAG]
    image_dimension = all_images.shape
    training_data = []
    for i in range(N_PATCHES_TRAIN):
        image_id = randint(image_dimension[2])
        patch_x_start = randint(image_dimension[0] - PATCH_SIZE[0])
        patch_x_end = patch_x_start + PATCH_SIZE[0]
        patch_y_start = randint(image_dimension[1] - PATCH_SIZE[1])
        patch_y_end = patch_y_start + PATCH_SIZE[1]

        # Slice out the patch
        patch = all_images[patch_x_start:patch_x_end,
                           patch_y_start:patch_y_end,
                           image_id]
        # Flatten the patch and append
        training_data.append(np.ravel(patch))

    if whitening:
        training_data -= np.mean(training_data)
        training_data /= np.std(training_data)
        training_data += recenter_std
        training_data /= recenter_std * 2

    return np.asarray(training_data)


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

#Global features classifier
def globalClassifier(globalD):
	
	covMat, mean = getCovarianceandMean(globalD)
	covMatInv = np.linalg.inv(covMat)

	maxThreshold = 0
	for x in globalD:#a row, note a patch
		threshold = np.dot(np.dot((x - mean), covMatInv),(x - mean).transpose())	
		if threshold > maxThreshold:
			maxThreshold = threshold

		# print np.dot((x - mean), covMatInv)
	print(maxThreshold)
	return maxThreshold, mean, covMatInv


class SparseAutoencoder(object):
    """Sparse autoencoder that performs unsupervised learning to extract
    interesting features.

    Parameters
    ----------
    input_size : Size of input layer
    hidden_size : Size of hidden layer
    L : Regularization strength for weight decay term
    rho : Sparsity parameter
    beta: Sparsity penalty weight

    Attributes
    ----------
    L, rho, beta : See above
    params : A NetworkParams object storing the parameters for the
    neural network
    """
    def __init__(self, dataset, input_size=25,  hidden_size=25, scenario='Dining'):
        self._input_size = input_size    # side length of sampled image patches
        self._hidden_size = hidden_size    # side length of representative image patches
        self.rho = 0.05   # desired average activation of hidden units
        self.lamda = 0.0001 # weight decay parameter
        self.beta = 3      # weight of sparsity penalty term
        self.max_iterations = 400    # number of optimization iterations
        self.learningRate = 0.5

        self.MeanCovarianceFileName = "MeanCovariance/n_MeanCovarianceFile" + scenario + ".npy"
        self.WeightsFileName = "Weights/n_WeightsFile" + scenario+ ".npy"

        self.thresholdFileName = "Threshold/n_thresholdFile" + scenario 
        self.thresholdFile = open(self.thresholdFileName, "wb")

        self.train_data=dataset['train_data']
        self.test_data=dataset['test_data']


    def train(self, input_data=None):
        if input_data is not None:
            inputNodes=input_data
        else:
            inputNodes=self.train_data

        numberOfNodes = inputNodes.shape[1]
        W1=self.W1
        W2=self.W2
        b1=self.b1
        b2=self.b2
        learningRate=self.learningRate
        lamda=self.lamda
        beta=self.beta
        rho=self.rho

        max_iterations=self.max_iterations

        oldCost = 0.0
        cost = 0.0
        
        hiddenLayer = self.sigmoid(np.dot(W1, inputNodes) + b1)
        rhoCap = np.sum(hiddenLayer, axis = 1)/numberOfNodes#mean for rows
        tempInputNodes = np.transpose(inputNodes)#transpose

        for iter in range(max_iterations):
            sumOfSquaresError = 0.0
            weightDecay = 0.0
            sparsityPenalty = 0.0
            for i in range(numberOfNodes):
                hiddenLayer = self.sigmoid(np.dot(W1, np.reshape(tempInputNodes[i], (-1, 1))) + b1)
                outputLayer = self.sigmoid(np.dot(W2, hiddenLayer) + b2)
                diff = outputLayer - np.reshape(tempInputNodes[i], (-1, 1))

                sumOfSquaresError += 0.5 * np.sum(np.multiply(diff, diff)) / tempInputNodes.shape[1]#1/2 * ||h(x)-y|| * ||h(x)-y||
                weightDecay += 0.5 * lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))#lamda/2 * sum of square of all the w(ji)
                sparsityPenalty   += beta * self.KLDivergence(rho, rhoCap)#incorporate the KL-divergence term into your derivative calculation
			
                KLDivGrad = beta * (-(rho / rhoCap) + ((1 - rho) / (1 - rhoCap)))

                errOut = np.multiply(diff, np.multiply(outputLayer, 1 - outputLayer))#err for output node, f'(z)=a(1-a) for sigmoid activation function
                errHid = np.multiply(np.dot(np.transpose(W2), errOut)  + np.transpose(np.matrix(KLDivGrad)), np.multiply(hiddenLayer, 1 - hiddenLayer))#err for hidden units, and incorporate the KL-divergence term into the derivate calculation
		
			    #Compute the gradient values by averaging partial derivatives
                W2Grad = np.dot(errOut, np.transpose(hiddenLayer))
                W1Grad = np.dot(errHid, np.transpose(np.reshape(tempInputNodes[i], (-1, 1))))
                b1Grad = np.sum(errHid, axis = 1)
                b2Grad = np.sum(errOut, axis = 1)

			    #Partial derivatives are averaged over all training examples

                W1Grad = learningRate*(W1Grad / tempInputNodes.shape[1] + lamda * W1)
                W2Grad = learningRate*(W2Grad / tempInputNodes.shape[1] + lamda * W2)
                b1Grad = learningRate*(b1Grad / tempInputNodes.shape[1])
                b2Grad = learningRate*(b2Grad / tempInputNodes.shape[1])	

                W1Grad = np.array(W1Grad)
                W2Grad = np.array(W2Grad)
                b1Grad = np.array(b1Grad)
                b2Grad = np.array(b2Grad)

			    # print b2Grad.shape, b2.shape
                W1 = W1 - W1Grad
                W2 = W2 - W2Grad
                b1 = b1 - b1Grad
                b2 = b2 - np.reshape(b2Grad, (-1, 1))

            oldCost = cost
            cost = sumOfSquaresError + weightDecay + sparsityPenalty
            if ((cost - oldCost)*(cost - oldCost) < 0.05):
                break

        globalD = np.dot(W1, inputNodes).transpose()
        threshold, mean, covMatInv = globalClassifier(globalD)

        self.thresholdFile.write("%f" % threshold)
        np.save(self.WeightsFileName, W1)
        np.savez(self.MeanCovarianceFileName, mean = mean, covMatInv = covMatInv)

    def predict(self, input_data=None):
        if input_data is not None:
            inputNodes=input_data
        else:
            inputNodes=self.test_data

        inputNodes = np.array(inputNodes, dtype = np.float32)  #Convert to np array
        inputNodes = np.transpose(inputNodes)
        inputNodes = (inputNodes - inputNodes.mean(axis=0)) / inputNodes.std(axis=0) #Normalization

        """Read Threshold"""
        thresholds = []
        directory = 'Threshold'
        thresholdFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
        thresholdFileNames.sort()

        for thresholdFileName in thresholdFileNames:
            thresholdFile = open(thresholdFileName, "rb")
            thresholds.append(float(thresholdFile.read()))

        """Read weights"""
        directory = 'Weights'
        weightFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
        weightFileNames.sort()

        """Read Covariance and Mean"""
        directory = 'MeanCovariance'
        meanCovarianceFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
        meanCovarianceFileNames.sort()

        for i in range(len(weightFileNames)):
            W1 = np.load(weightFileNames[i])
            data = np.load(meanCovarianceFileNames[i])
            mean = data['mean']
            covMatInv = data['covMatInv']

        globalD = np.dot(W1, inputNodes).transpose()

    def init_params(self):
        r = np.sqrt(2.0)/np.sqrt(self._input_size + self._hidden_size)
        rand = np.random.RandomState(int(time.time()))
        self.W1 = np.array(rand.uniform(low = -r, high = r, size = (self._hidden_size, self._input_size)))
        self.W2 = np.array(rand.uniform(low = -r, high = r, size = (self._input_size, self._hidden_size)))
        self.b1 = np.zeros((self._hidden_size, 1))
        self.b2 = np.zeros((self._input_size, 1))

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def KLDivergence(self, rho, rhoCap):
    	# print rhoCap
	    return np.sum(rho*np.log(rho/rhoCap) + (1 - rho)*np.log((1 - rho)/ (1 - rhoCap)))